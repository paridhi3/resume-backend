embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
NotImplementedError: Cannot copy out of meta tensor; no data! Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to() when moving module from meta to a different device.

File "C:\Users\703417007_agarwal\Desktop\resume-backend\venv\Lib\site-packages\streamlit\runtime\scriptrunner\exec_code.py", line 128, in exec_func_with_error_handling
    result = func()
File "C:\Users\703417007_agarwal\Desktop\resume-backend\venv\Lib\site-packages\streamlit\runtime\scriptrunner\script_runner.py", line 669, in code_to_exec
    exec(code, module.__dict__)  # noqa: S102
    ~~~~^^^^^^^^^^^^^^^^^^^^^^^
File "C:\Users\703417007_agarwal\Desktop\resume-backend\llm_app.py", line 24, in <module>
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
File "C:\Users\703417007_agarwal\Desktop\resume-backend\venv\Lib\site-packages\sentence_transformers\SentenceTransformer.py", line 367, in __init__
    self.to(device)
    ~~~~~~~^^^^^^^^
File "C:\Users\703417007_agarwal\Desktop\resume-backend\venv\Lib\site-packages\torch\nn\modules\module.py", line 1369, in to
    return self._apply(convert)
           ~~~~~~~~~~~^^^^^^^^^
File "C:\Users\703417007_agarwal\Desktop\resume-backend\venv\Lib\site-packages\torch\nn\modules\module.py", line 928, in _apply
    module._apply(fn)
    ~~~~~~~~~~~~~^^^^
File "C:\Users\703417007_agarwal\Desktop\resume-backend\venv\Lib\site-packages\torch\nn\modules\module.py", line 928, in _apply
    module._apply(fn)
    ~~~~~~~~~~~~~^^^^
File "C:\Users\703417007_agarwal\Desktop\resume-backend\venv\Lib\site-packages\torch\nn\modules\module.py", line 928, in _apply
    module._apply(fn)
    ~~~~~~~~~~~~~^^^^
[Previous line repeated 1 more time]
File "C:\Users\703417007_agarwal\Desktop\resume-backend\venv\Lib\site-packages\torch\nn\modules\module.py", line 955, in _apply
    param_applied = fn(param)
File "C:\Users\703417007_agarwal\Desktop\resume-backend\venv\Lib\site-packages\torch\nn\modules\module.py", line 1362, in convert
    raise NotImplementedError(
    ...<2 lines>...
    ) from None