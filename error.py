OSError: [E053] Could not read config file from C:\Users\703417007_agarwal\Desktop\resume-backend\venv\Lib\site-packages\pyresparser\config.cfg

File "C:\Users\703417007_agarwal\Desktop\resume-backend\venv\Lib\site-packages\streamlit\runtime\scriptrunner\exec_code.py", line 128, in exec_func_with_error_handling
    result = func()
File "C:\Users\703417007_agarwal\Desktop\resume-backend\venv\Lib\site-packages\streamlit\runtime\scriptrunner\script_runner.py", line 669, in code_to_exec
    exec(code, module.__dict__)  # noqa: S102
    ~~~~^^^^^^^^^^^^^^^^^^^^^^^
File "C:\Users\703417007_agarwal\Desktop\resume-backend\app2.py", line 278, in <module>
    parsed = process_and_save_resume(resume_file, resume_text, metadata_path="metadata.json")
File "C:\Users\703417007_agarwal\Desktop\resume-backend\app2.py", line 94, in process_and_save_resume
    parsed_data = ResumeParser(file_path).get_extracted_data()
                  ~~~~~~~~~~~~^^^^^^^^^^^
File "C:\Users\703417007_agarwal\Desktop\resume-backend\venv\Lib\site-packages\pyresparser\resume_parser.py", line 21, in __init__
    custom_nlp = spacy.load(os.path.dirname(os.path.abspath(__file__)))
File "C:\Users\703417007_agarwal\Desktop\resume-backend\venv\Lib\site-packages\spacy\__init__.py", line 52, in load
    return util.load_model(
           ~~~~~~~~~~~~~~~^
        name,
        ^^^^^
    ...<4 lines>...
        config=config,
        ^^^^^^^^^^^^^^
    )
    ^
File "C:\Users\703417007_agarwal\Desktop\resume-backend\venv\Lib\site-packages\spacy\util.py", line 479, in load_model
    return load_model_from_path(Path(name), **kwargs)  # type: ignore[arg-type]
File "C:\Users\703417007_agarwal\Desktop\resume-backend\venv\Lib\site-packages\spacy\util.py", line 550, in load_model_from_path
    config = load_config(config_path, overrides=overrides)
File "C:\Users\703417007_agarwal\Desktop\resume-backend\venv\Lib\site-packages\spacy\util.py", line 726, in load_config
    raise IOError(Errors.E053.format(path=config_path, name="config file"))