"""Pytest configuration to inject lightweight fakes for heavy optional deps.

This helps run the full test suite (collection phase) on low-resource
environments by providing minimal APIs for packages like `datasets`,
`transformers`, `torch`, `faiss`, `moviepy`, `gtts`, and various
`langchain` packages. Individual tests may still monkeypatch more
specific behavior as needed.
"""
import sys
import types


def _make_module(name: str):
    m = types.ModuleType(name)
    return m


# Minimal `datasets` shim with load_dataset that returns a dict-like
def _datasets_load_dataset(name):
    class SimpleSplit:
        def __init__(self):
            self._data = [{"text": "Example review 1"}, {"text": "Example review 2"}]

        def shuffle(self, seed=None):
            return self

        def select(self, rng):
            return self._data[: len(rng)]

        def __iter__(self):
            return iter(self._data)

        def __getitem__(self, key):
            # allow imdb['train'] usage
            if key == "train":
                return self
            raise KeyError(key)

    return {"train": SimpleSplit()}


# Minimal `transformers` shim with pipeline
def _transformers_pipeline(task: str, **kwargs):
    task = (task or "").lower()

    if "summar" in task:
        def _summarize(text, max_length=None, min_length=None, do_sample=False, **call_kwargs):
            # produce a fake summary with a word count between min_length and max_length
            try:
                min_l = int(min_length) if min_length else 10
                max_l = int(max_length) if max_length else min_l + 10
            except Exception:
                min_l = 10
                max_l = min_l + 10
            target = min_l + max(0, (max_l - min_l) // 2)
            # If the original text mentions notable phrases, include them
            kws = []
            txt_low = text.lower() if isinstance(text, str) else " ".join(text).lower()
            if "hugging face" in txt_low:
                kws.append("Hugging Face")
            for term in ["machine learning", "chatbot", "transformers", "new york"]:
                if term in txt_low:
                    kws.append(term)

            filler_count = max(0, target - len(kws))
            words = kws + ["summaryword"] * filler_count
            return [{"summary_text": " ".join(words)}]

        return _summarize

    if "sentiment" in task:
        def _sentiment(text, truncation=True, max_length=None, **call_kwargs):
            txt = text.lower() if isinstance(text, str) else " ".join(text)
            if any(w in txt for w in ["unclear", "wasted", "bad", "hate", "terrible", "sad"]):
                return [{"label": "NEGATIVE", "score": 0.9}]
            return [{"label": "POSITIVE", "score": 0.99}]

        return _sentiment

    # generic generation
    def _gen(text, **call_kwargs):
        return [{"generated_text": text + "...generated"}]

    return _gen


def _install_shims():
    # datasets
    ds = _make_module("datasets")
    ds.load_dataset = _datasets_load_dataset
    sys.modules["datasets"] = ds

    # transformers
    tf = _make_module("transformers")
    tf.pipeline = _transformers_pipeline
    sys.modules["transformers"] = tf

    # lightweight torch shim
    sys.modules.setdefault("torch", _make_module("torch"))

    # faiss shim
    sys.modules.setdefault("faiss", _make_module("faiss"))

    # moviepy shim
    mp = _make_module("moviepy")
    mp.editor = _make_module("moviepy.editor")
    sys.modules["moviepy"] = mp
    sys.modules["moviepy.editor"] = mp.editor
    # Provide top-level names some examples import
    class _ImageSequenceClip:
        def __init__(self, frames, fps=24):
            self.frames = frames

        def write_videofile(self, path, codec=None):
            open(path, "wb").write(b"fake-video")

    class _AudioFileClip:
        def __init__(self, path):
            self.path = path

        def write_audiofile(self, path):
            open(path, "wb").write(b"fake-audio")

    mp.ImageSequenceClip = _ImageSequenceClip
    mp.AudioFileClip = _AudioFileClip

    # gtts shim
    gtts = _make_module("gtts")
    class _G:
        def __init__(self, text, lang="en"):
            self.text = text

        def save(self, path):
            with open(path, "wb") as f:
                f.write(b"fake-audio")

    gtts.gTTS = _G
    sys.modules["gtts"] = gtts

    # Provide a minimal langchain_openai shim only. Tests that need a
    # different langchain implementation should monkeypatch sys.modules
    # themselves to control which one is imported.
    class FakeChatOpenAI:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, prompt):
            return f"fake-chat: {prompt}"

    lco = _make_module("langchain_openai")
    lco.ChatOpenAI = FakeChatOpenAI
    sys.modules["langchain_openai"] = lco

    # sqlmodel shim used by some examples/tests
    sm = _make_module("sqlmodel")
    # in-memory DB used by the fake Session
    global _IN_MEMORY_DB
    _IN_MEMORY_DB = {}
    # Make SQLModel behave like a Pydantic model so FastAPI can create
    # response models from it during route registration in tests.
    try:
        from pydantic import BaseModel as _PydBase

        class SQLModel(_PydBase):
            def __init_subclass__(cls, **kwargs):
                # Accept kwargs like table=True from datamodels
                super().__init_subclass__()
    except Exception:
        class SQLModel:
            def __init_subclass__(cls, **kwargs):
                super().__init_subclass__()

    class Session:
        def __init__(self, *args, **kwargs):
            # Simple in-memory store shared across fake sessions
            self._store = _IN_MEMORY_DB

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def add(self, obj):
            # assign a simple integer id and store the dict form keyed by model name
            model_name = getattr(obj.__class__, "__name__", "item").lower()
            key = model_name + "s"
            lst = self._store.setdefault(key, [])
            new_id = len(lst) + 1
            try:
                obj.id = new_id
                record = obj.model_dump() if hasattr(obj, "model_dump") else obj.__dict__.copy()
            except Exception:
                record = getattr(obj, "__dict__", {}).copy()
                record["id"] = new_id
            record.setdefault("id", new_id)
            lst.append(record)

        def commit(self):
            return None

        def refresh(self, obj):
            # populate obj.id if missing
            return None

        def exec(self, query):
            # return an object with all() that gives the stored tasks
            class _Q:
                def all(inner_self):
                    # If query contains a model class, try to map to storage key
                    try:
                        model = query[0]
                        key = model.__name__.lower() + "s"
                        items = self._store.get(key, [])
                        # Try to return instances of model when possible
                        try:
                            return [model(**it) if isinstance(it, dict) else it for it in items]
                        except Exception:
                            return items
                    except Exception:
                        # Fallback: return all stored items concatenated
                        out = []
                        for v in self._store.values():
                            out.extend(v)
                        return out

            return _Q()

        def get(self, model, id):
            key = model.__name__.lower() + "s"
            for rec in self._store.get(key, []):
                if rec.get("id") == id:
                    try:
                        return model(**rec)
                    except Exception:
                        return rec
            return None

        def delete(self, obj):
            # obj may be a dict with an 'id' and we infer collection by type name
            target_id = None
            if isinstance(obj, dict):
                target_id = obj.get("id")
            elif isinstance(obj, int):
                target_id = obj
            else:
                try:
                    target_id = getattr(obj, "id", None)
                except Exception:
                    target_id = None

            if target_id is None:
                return None

            removed = None
            for key, lst in list(self._store.items()):
                new_lst = [rec for rec in lst if rec.get("id") != target_id]
                if len(new_lst) != len(lst):
                    # something was removed
                    removed = True
                    self._store[key] = new_lst
            return removed

    def create_engine(*args, **kwargs):
        return None

    def select(*args, **kwargs):
        return args

    try:
        from pydantic import Field as _PydField

        def Field(default=None, *args, **kwargs):
            return _PydField(default=default, *args, **kwargs)
    except Exception:
        def Field(default=None, *args, **kwargs):
            return default

    sm.SQLModel = SQLModel
    sm.Session = Session
    sm.create_engine = create_engine
    sm.select = select
    sm.Field = Field
    sys.modules["sqlmodel"] = sm


# Install shims early during collection
_install_shims()
