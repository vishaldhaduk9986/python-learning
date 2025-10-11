import sys
import types
import runpy
import os
import io
import contextlib
import importlib

from pathlib import Path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
import pytest

MODULES_TO_TEST = [

# Ensure repo root is on sys.path so imports like `importlib.import_module('src.day15')`
# work when running a single test or invoking pytest from a subdirectory.
    'src.day15',
    'src.day16',
    'src.day17',
    'src.day18',
    'src.day19',
    'src.day20',
    'src.day21',
]


class DummyLLM:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, prompt):
        return 'dummy-response'

    def invoke(self, prompt):
        return 'dummy-response'


class DummyLLMChain:
    def __init__(self, *args, **kwargs):
        pass

    def run(self, *args, **kwargs):
        # For SequentialChain which returns dict-like, return a string or dict if requested
        return 'dummy-run'


class DummySequentialChain(DummyLLMChain):
    def run(self, *args, **kwargs):
        return {'summary': 'dummy-summary', 'keywords': 'dummy-keywords'}
    def __call__(self, *args, **kwargs):
        # Allow the chain to be invoked like a callable with an input dict
        return self.run(*args, **kwargs)


class DummyPromptTemplate:
    def __init__(self, input_variables=None, template=''):
        self.input_variables = input_variables
        self.template = template
    @classmethod
    def from_template(cls, template_str: str):
        # Return an instance similar to the real PromptTemplate.from_template
        return cls(input_variables=[], template=template_str)


class DummyLoader:
    def __init__(self, path):
        self.path = path

    def load(self):
        class Doc:
            def __init__(self, content, path):
                self.page_content = content
                self.metadata = {'source': path}

        # Return a small list of docs
        content = "sample content from %s" % self.path
        return [Doc(content, self.path)]

    def load_and_split(self):
        return self.load()


class DummyTextLoader(DummyLoader):
    pass


class DummyEmbeddings:
    def __init__(self, *args, **kwargs):
        pass

    def embed_documents(self, texts):
        # return dummy vectors
        return [[0.1] * 3 for _ in texts]


class DummyFAISS:
    def __init__(self):
        self.docs = []

    @classmethod
    def from_documents(cls, docs, embeddings):
        inst = cls()
        inst.docs = docs
        return inst

    def as_retriever(self):
        return self

    def similarity_search(self, query, k=3):
        return self.docs[:k]


@pytest.fixture(autouse=True)
def inject_fakes(tmp_path, monkeypatch):
    """Insert fake langchain-related modules into sys.modules before imports."""
    fake_modules = {}

    # Create package modules and submodules
    def make_mod(name):
        m = types.ModuleType(name)
        fake_modules[name] = m
        return m

    # langchain_community.llms and langchain.llms
    lc_community_llms = make_mod('langchain_community.llms')
    lc_llms = make_mod('langchain.llms')
    lc_openai = make_mod('langchain_openai')

    # classes
    lc_community_llms.ChatOpenAI = DummyLLM
    lc_llms.ChatOpenAI = DummyLLM
    lc_openai.ChatOpenAI = DummyLLM
    # langchain_openai may also expose OpenAIEmbeddings
    lc_openai.OpenAIEmbeddings = DummyEmbeddings

    # chains
    lc_chains = make_mod('langchain.chains')
    lc_chains.LLMChain = DummyLLMChain
    lc_chains.SequentialChain = DummySequentialChain
    # RetrievalQA is used in day20/day21
    class DummyRetrievalQA:
        def __init__(self, *args, **kwargs):
            pass
        def run(self, q):
            return 'dummy-retrieval-answer'
    lc_chains.RetrievalQA = DummyRetrievalQA
    # create_retrieval_chain should return an object with an `invoke` method
    def fake_create_retrieval_chain(retriever, combine_chain):
        class FakeChain:
            def invoke(self, payload):
                # Return a dict-like result similar to langchain
                return {"answer": "fake-answer", "source_documents": []}
        return FakeChain()

    lc_chains.create_retrieval_chain = fake_create_retrieval_chain

    # create_stuff_documents_chain is expected to accept llm and prompt
    # and return a chain-like object (we'll return a simple object)
    combine_mod = make_mod('langchain.chains.combine_documents')
    def fake_create_stuff_documents_chain(llm, prompt):
        class CombineChain:
            def run(self, *args, **kwargs):
                return {"answer": "combined-fake", "source_documents": []}
        return CombineChain()

    combine_mod.create_stuff_documents_chain = fake_create_stuff_documents_chain

    # prompts
    lc_prompts = make_mod('langchain.prompts')
    lc_prompts.PromptTemplate = DummyPromptTemplate
    # Some files import PromptTemplate from langchain_core.prompts
    lc_core_prompts = make_mod('langchain_core.prompts')
    lc_core_prompts.PromptTemplate = DummyPromptTemplate

    # community document loaders and text loaders
    lc_comm_doc = make_mod('langchain_community.document_loaders')
    lc_doc = make_mod('langchain.document_loaders')
    lc_comm_doc.PyPDFLoader = DummyLoader
    lc_doc.PyPDFLoader = DummyLoader
    # TextLoader
    lc_comm_doc.TextLoader = DummyTextLoader

    # embeddings
    lc_comm_emb = make_mod('langchain_community.embeddings')
    lc_emb = make_mod('langchain.embeddings')
    lc_comm_emb.OpenAIEmbeddings = DummyEmbeddings
    # Create a proper submodule langchain.embeddings.openai so imports like
    # `from langchain.embeddings.openai import OpenAIEmbeddings` work.
    emb_openai = make_mod('langchain.embeddings.openai')
    emb_openai.OpenAIEmbeddings = DummyEmbeddings

    # vectorstores
    lc_comm_vs = make_mod('langchain_community.vectorstores')
    lc_vs = make_mod('langchain.vectorstores')
    lc_comm_vs.FAISS = DummyFAISS
    lc_vs.FAISS = DummyFAISS

    # text_splitter
    lc_text_splitter = make_mod('langchain.text_splitter')
    class DummySplitter:
        def __init__(self, chunk_size=1000, chunk_overlap=0):
            pass
        def split_documents(self, docs):
            return docs
    lc_text_splitter.CharacterTextSplitter = DummySplitter

    # put them into sys.modules
    for k, v in fake_modules.items():
        monkeypatch.setitem(sys.modules, k, v)

    # Ensure sample text files exist for day18 (file1..file10)
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    for i in range(1, 11):
        p = os.path.join(root, f'file{i}.txt')
        if not os.path.exists(p):
            with open(p, 'w') as f:
                f.write(f"Sample content for file {i}\n")

    yield


def run_module_and_capture(name):
    # run module as script and capture stdout
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            runpy.run_module(name, run_name='__main__')
        except SystemExit as e:
            # capture SystemExit code
            print(f"SystemExit:{e.code}")
        except Exception as e:
            print(f"Exception:{e}")
    return buf.getvalue()


@pytest.mark.parametrize('modname', MODULES_TO_TEST)
def test_modules_run_without_errors(modname):
    out = run_module_and_capture(modname)
    # Check that output contains some expected marker or not an exception
    assert 'Exception:' not in out
