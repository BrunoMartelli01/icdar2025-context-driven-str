"""Native Long-CLIP/LLM2CLIP inference and a versioned cross-environment bridge.

This module deliberately imports no evaluation, recognition, or training stack.
The native worker can run in the existing LAVIS environment for Long-CLIP.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import random
import subprocess
import sys
import tempfile
import traceback
import types

import numpy as np
import torch
from PIL import Image


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_files(identity):
    for name, digest in identity['file_sha256'].items():
        if file_sha(name) != digest:
            raise ValueError('Pinned artifact changed: ' + name)


def module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def longclip_dependency_compatibility(source):
    """Replace only the removed setuptools re-export, not native model code."""
    old = 'from pkg_resources import packaging\n'
    if source.count(old) != 1:
        raise ValueError('Pinned Long-CLIP packaging import changed')
    return source.replace(old, 'import packaging.version\n', 1)


def load_longclip_native(name, source):
    package = types.ModuleType(name)
    package.__path__ = [str(source)]
    sys.modules[name] = package
    path = source / 'longclip.py'
    spec = importlib.util.spec_from_file_location(name + '.longclip', path)
    native = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = native
    exec(compile(longclip_dependency_compatibility(path.read_text()), str(path), 'exec'), native.__dict__)
    return native


def llama_caption_parts(text):
    """LLM2Vec no-instruction Llama3 framing, excluding the role prefix in pooling.

    LLM2Vec first prefixes its instruction delimiter, then strips the framed
    string. Consequently leading body whitespace survives and trailing body
    whitespace is stripped. The delimiter is never sent to the model.
    """
    if '!@#$%^&*()' in text:
        raise ValueError('Raw transcript contains the native LLM2Vec delimiter')
    body = text.rstrip() + '<|eot_id|>'
    return '<|start_header_id|>user<|end_header_id|>\n\n' + body, body


def caption_mean_pool(hidden, lengths):
    if len(lengths) != len(hidden) or any(n <= 0 or n > hidden.shape[1] for n in lengths):
        raise ValueError('Invalid native left-padded pooling lengths')
    return torch.stack([hidden[i, -n:, :].mean(dim=0) for i, n in enumerate(lengths)])


def normalized_vectors(value):
    value = torch.nn.functional.normalize(value.float(), dim=-1)
    if not torch.isfinite(value).all() or torch.any(value.norm(dim=-1) == 0):
        raise ValueError('Invalid native embedding')
    return value.cpu().numpy().astype(np.float32, copy=False)


def clip_rgb_pixels(processor, images):
    # The bridge guarantees HWC RGB. Guessing from a 3x3 crop is ambiguous,
    # and a guessed channel axis can corrupt a batch or reject its next crop.
    return processor(images=images, return_tensors='pt',
                     input_data_format='channels_last')['pixel_values']


class NativeEncoder:
    def __init__(self, identity, device):
        self.identity, self.device = identity, device
        self.batch_size = identity['batch_size']
        self.limit = identity['context_length']
        self.backend = identity['backend']
        self.dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16}[identity['precision']]
        verify_files(identity)
        if device.startswith('cuda'):
            if not torch.cuda.is_available():
                raise RuntimeError('Requested CUDA is unavailable')
            torch.cuda.init()
            torch.cuda.reset_peak_memory_stats(device)
        if self.backend == 'longclip_native':
            source = Path(identity['source_root']) / 'model'
            name = 'qualified_longclip_' + identity['source_revision'][:12]
            self.native = load_longclip_native(name, source)
            # Same published builder and strict loading, with safe tensor-only deserialization.
            state = torch.load(identity['checkpoint'], map_location='cpu', weights_only=True)
            original_keys = set(state) - {'input_resolution', 'context_length', 'vocab_size'}
            self.model = self.native.build_model(state, load_from_clip=False)
            if set(self.model.state_dict()) != original_keys:
                raise ValueError('Native Long-CLIP checkpoint keys differ')
            self.model = self.model.eval().to(dtype=self.dtype, device=device)
            self.processor = self.native._transform(self.model.visual.input_resolution)
            if (self.model.context_length != self.limit
                    or self.model.visual.input_resolution != identity['image_size']):
                raise ValueError('Published Long-CLIP dimensions disagree')
            self.load_info = {'missing_keys': [], 'unexpected_keys': [], 'mismatched_keys': []}
        elif self.backend == 'llm2clip_native':
            from transformers import AutoConfig, AutoTokenizer, CLIPImageProcessor
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
            vision = identity['vision_directory']; text = identity['text_directory']
            cls = get_class_from_dynamic_module('modeling_clip.LLM2CLIPModel', vision, local_files_only=True)
            # Re-importing the dynamic config independently can create a distinct
            # Python class object and fail the native isinstance guard.
            config_cls = cls.config_class
            config = config_cls.from_pretrained(vision, local_files_only=True)
            self.model, vision_info = cls.from_pretrained(vision, config=config,
                torch_dtype=self.dtype, local_files_only=True, output_loading_info=True,
                low_cpu_mem_usage=True, attn_implementation=identity['attention_implementation'])
            name = 'qualified_llm2clip_text_' + identity['text_revision'][:12]
            package = types.ModuleType(name); package.__path__ = [text]; sys.modules[name] = package
            module_from_path(name + '.attn_mask_utils', identity['mask_source'])
            native = module_from_path(name + '.modeling_llama_encoder', Path(text) / 'modeling_llama_encoder.py')
            text_config = AutoConfig.from_pretrained(text, local_files_only=True)
            self.text_model, text_info = native.LlamaEncoderModel.from_pretrained(text,
                config=text_config, torch_dtype=self.dtype, local_files_only=True,
                output_loading_info=True, low_cpu_mem_usage=True,
                attn_implementation=identity['attention_implementation'])
            for info in (vision_info, text_info):
                if any(info.values()):
                    raise ValueError('Native checkpoint incompatibility: ' + str(info))
            self.load_info = {'vision': vision_info, 'text': text_info}
            self.model = self.model.eval().to(device)
            self.text_model = self.text_model.eval().to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(text, local_files_only=True, use_fast=True)
            self.tokenizer.padding_side = 'left'
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.processor = CLIPImageProcessor.from_pretrained(identity['processor_directory'], local_files_only=True)
            if self.model.config.vision_config.image_size != identity['image_size']:
                raise ValueError('Native LLM2CLIP input resolution disagrees')
        else:
            raise ValueError('Unsupported native encoder backend')

    def token_ids(self, text):
        if self.backend == 'longclip_native':
            tokenizer = self.native._tokenizer
            return [tokenizer.encoder['<|startoftext|>'], *tokenizer.encode(text), tokenizer.encoder['<|endoftext|>']]
        framed, _ = llama_caption_parts(text)
        return self.tokenizer(framed, add_special_tokens=True, truncation=False)['input_ids']

    def embed(self, values, images=False):
        outputs = []
        for start in range(0, len(values), self.batch_size):
            batch = values[start:start + self.batch_size]
            with torch.inference_mode():
                if images:
                    if self.backend == 'longclip_native':
                        pixels = torch.stack([self.processor(x) for x in batch]).to(self.device, self.dtype)
                        result = self.model.encode_image(pixels)
                    else:
                        pixels = clip_rgb_pixels(self.processor,batch).to(self.device, self.dtype)
                        with torch.autocast(device_type=torch.device(self.device).type, dtype=self.dtype):
                            result = self.model.get_image_features(pixels)
                elif self.backend == 'longclip_native':
                    if any(len(self.token_ids(t)) > self.limit for t in batch):
                        raise ValueError('Refusing native text truncation')
                    tokens = self.native.tokenize(batch, context_length=self.limit, truncate=False).to(self.device)
                    result = self.model.encode_text(tokens)
                else:
                    parts = [llama_caption_parts(t) for t in batch]
                    if any(len(self.token_ids(t)) > self.limit for t in batch):
                        raise ValueError('Refusing native text truncation')
                    tokens = self.tokenizer([p[0] for p in parts], padding=True,
                        return_tensors='pt', truncation=False)
                    lengths = [len(self.tokenizer(p[1], add_special_tokens=False,
                        truncation=False)['input_ids']) for p in parts]
                    hidden = self.text_model(**{k:v.to(self.device) for k,v in tokens.items()},
                        use_cache=False).last_hidden_state
                    # Published LLM2Vec.encode returns FP32 before the trained adapter.
                    pooled = caption_mean_pool(hidden, lengths).float()
                    with torch.autocast(device_type=torch.device(self.device).type, dtype=self.dtype):
                        result = self.model.get_text_features(pooled)
                outputs.extend(normalized_vectors(result))
        return np.asarray(outputs, dtype=np.float32)

    def memory(self):
        cuda = self.device.startswith('cuda')
        return {'peak_allocated_bytes': torch.cuda.max_memory_allocated(self.device) if cuda else None,
                'peak_reserved_bytes': torch.cuda.max_memory_reserved(self.device) if cuda else None,
                'peak_cpu_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}


def worker(identity_path, device):
    identity = json.loads(Path(identity_path).read_text())
    torch.set_num_threads(identity['cpu_threads']); torch.manual_seed(identity['seed'])
    random.seed(identity['seed']); np.random.seed(identity['seed'])
    if device.startswith('cuda'):
        torch.cuda.manual_seed_all(identity['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        encoder = NativeEncoder(identity, device)
    except Exception as error:
        cuda = device.startswith('cuda') and torch.cuda.is_available()
        print(json.dumps({'schema_version':1,'success':False,'phase':'model_loading',
            'error':str(error),'traceback':traceback.format_exc(),
            'out_of_memory':'out of memory' in str(error).lower(),
            'memory':{'peak_allocated_bytes':torch.cuda.max_memory_allocated(device) if cuda else None,
                'peak_reserved_bytes':torch.cuda.max_memory_reserved(device) if cuda else None}}), flush=True)
        raise
    import importlib.metadata
    software = {'torch':torch.__version__,'python':sys.version.split()[0],
        'numpy':np.__version__,'pillow':__import__('PIL').__version__,
        'transformers':importlib.metadata.version('transformers'),
        'torchvision':importlib.metadata.version('torchvision')}
    if encoder.backend == 'longclip_native':
        software['ftfy'] = importlib.metadata.version('ftfy')
    print(json.dumps({'schema_version':1,'success':True,'ready':True,'load_validation':encoder.load_info,
        'software':software,'memory':encoder.memory()}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        try:
            if request['schema_version'] != 1:
                raise ValueError('Unsupported native bridge schema')
            if request['op'] == 'close':
                print(json.dumps({'schema_version':1,'success':True,'closed':True}), flush=True)
                return
            if request['op'] == 'token_ids':
                response = {'input_ids':encoder.token_ids(request['text'])}
            else:
                if request['op'] == 'images':
                    values = []
                    for source in request['images']:
                        path = Path(source['path'])
                        if file_sha(path) != source['sha256']:
                            raise ValueError('Bridge pixel payload changed')
                        pixels = np.load(path, allow_pickle=False)
                        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[-1] != 3:
                            raise ValueError('Expected RGB uint8 bridge input')
                        values.append(Image.fromarray(pixels, 'RGB'))
                elif request['op'] == 'texts':
                    values = request['texts']
                else:
                    raise ValueError('Unsupported native operation')
                vectors = encoder.embed(values, images=request['op']=='images')
                path = Path(request['output_path']); np.save(path, vectors, allow_pickle=False)
                response = {'path':str(path),'sha256':file_sha(path),'shape':list(vectors.shape),'dtype':str(vectors.dtype)}
            print(json.dumps({'schema_version':1,'success':True,'request_id':request['request_id'],
                'memory':encoder.memory(), **response}), flush=True)
        except Exception as error:
            print(json.dumps({'schema_version':1,'success':False,'error':str(error),
                'traceback':traceback.format_exc(),'out_of_memory':isinstance(error,MemoryError)
                or 'out of memory' in str(error).lower(),'memory':encoder.memory()}), flush=True)
            raise


class RemoteTokenizer:
    def __init__(self, bridge):
        self.bridge, self.cache = bridge, {}

    def __call__(self, text, *, add_special_tokens, truncation):
        if not isinstance(text,str) or add_special_tokens is not True or truncation is not False:
            raise ValueError('Native length checks require untruncated, special-token input')
        if text not in self.cache:
            self.cache[text] = self.bridge.request({'op':'token_ids','text':text})['input_ids']
        return {'input_ids':self.cache[text]}


class EncoderBridge:
    def __init__(self, identity, device, root, python):
        self.identity, self.device = identity, device
        self.limit = identity['context_length']; self.cache = {}; self.counter = 0
        root = Path(root); root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='native-', dir=root)
        self.directory = Path(self.temporary.name)
        path = self.directory / 'identity.json'; path.write_text(json.dumps(identity,indent=2)+'\n')
        self.log = (root/'native_worker.stderr.log').open('a')
        env = os.environ.copy(); env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        self.process = subprocess.Popen([str(python),'-m','scripts.clip_specialized_worker',
            '--identity',str(path),'--device',device],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=self.log,text=True,bufsize=1,env=env)
        self.failed = False
        try:
            self.ready = self.read_response()
        except Exception:
            if self.process.poll() is None:
                self.process.terminate(); self.process.wait(timeout=30)
            self.log.close(); self.temporary.cleanup()
            raise
        self.memory = self.ready['memory']; self.tokenizer = RemoteTokenizer(self)

    def read_response(self):
        line = self.process.stdout.readline()
        if not line:
            self.failed = True
            raise RuntimeError('Native worker exited; inspect its stderr log')
        response = json.loads(line)
        if 'memory' in response:
            self.memory = response['memory']
        if response.get('schema_version') != 1 or response.get('success') is not True:
            self.failed = True
            raise RuntimeError('Native inference failed: '+json.dumps(response))
        return response

    def request(self, request):
        self.counter += 1
        request.update(schema_version=1,request_id=self.counter)
        self.process.stdin.write(json.dumps(request)+'\n'); self.process.stdin.flush()
        response = self.read_response()
        if response.get('request_id') != self.counter:
            self.failed = True
            raise ValueError('Native bridge response ordering changed')
        self.memory = response['memory']
        return response

    def embeddings(self, values, images=False):
        files = []
        output = self.directory/'vectors.npy'
        try:
            request = {'op':'images' if images else 'texts','output_path':str(output)}
            if images:
                for index, value in enumerate(values):
                    if not isinstance(value,Image.Image) or value.mode != 'RGB':
                        raise ValueError('Expected original RGB PIL input')
                    path = self.directory/('pixels-'+str(index)+'.npy')
                    np.save(path,np.asarray(value),allow_pickle=False);files.append(path)
                request['images'] = [{'path':str(p),'sha256':file_sha(p)} for p in files]
            else:
                request['texts'] = values
            response = self.request(request)
            if response['path'] != str(output) or file_sha(output) != response['sha256']:
                raise ValueError('Native embedding payload changed')
            vectors = np.load(output,allow_pickle=False)
            if (vectors.dtype != np.float32 or list(vectors.shape) != response['shape']
                    or len(vectors) != len(values) or not np.isfinite(vectors).all()
                    or not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-5)):
                raise ValueError('Invalid native embedding population')
            return vectors
        finally:
            for path in [*files, output]:
                path.unlink(missing_ok=True)

    def ensure(self, prompts):
        missing = sorted(set(prompts)-self.cache.keys())
        if missing:
            self.cache.update(zip(missing,self.embeddings(missing)))

    def mean(self, prompts):
        vector = np.stack([self.cache[p] for p in prompts]).mean(axis=0)
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError('Invalid native prompt ensemble')
        return vector/norm

    def close(self):
        try:
            if self.process.poll() is None:
                if self.failed:
                    self.process.terminate(); self.process.wait(timeout=30)
                else:
                    self.process.stdin.write(json.dumps({'schema_version':1,'op':'close'})+'\n')
                    self.process.stdin.flush(); self.read_response()
                    self.process.wait(timeout=30)
            if self.process.returncode != 0 and not self.failed:
                raise RuntimeError('Native worker exited with a nonzero status')
        finally:
            if self.process.poll() is None:
                self.process.terminate(); self.process.wait(timeout=30)
            self.log.close(); self.temporary.cleanup()
