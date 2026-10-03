import inspect
from models import common

print("="*60)
print("CLASS: Conv")
print("="*60)
try:
    print(inspect.getsource(common.Conv))
except Exception as e:
    print(e)

print("="*60)
print("CLASS: Bottleneck")
print("="*60)
try:
    print(inspect.getsource(common.Bottleneck))
except Exception as e:
    print(e)

print("="*60)
print("CLASS: Bottleneck_merged")
print("="*60)
try:
    print(inspect.getsource(common.Bottleneck_merged))
except Exception as e:
    print(e)