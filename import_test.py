import importlib, traceback, os, sys
print('cwd:', os.getcwd())
print('python executable:', sys.executable)
print('sys.path[0]:', sys.path[0])
print('listing cwd:')
for f in os.listdir('.'):
    print('  ', f)
try:
    m = importlib.import_module('train_model')
    print('Imported train_model from', getattr(m,'__file__','<unknown>'))
    print('has load_data?', hasattr(m,'load_data'))
    if hasattr(m,'load_data'):
        print('EMOTIONS:', getattr(m,'EMOTIONS', None))
        print('IMG_SIZE:', getattr(m,'IMG_SIZE', None))
except Exception:
    traceback.print_exc()
