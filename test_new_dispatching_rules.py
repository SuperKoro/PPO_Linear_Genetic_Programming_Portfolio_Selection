# test_new_dispatching_rules.py
"""
Script để test các dispatching rules mới đã được thêm vào.
"""

print("=" * 60)
print("Testing New Dispatching Rules Registration")
print("=" * 60)

# Import để trigger auto-registration
import dispatching_rules
from dispatching_registry import DR_REGISTRY, has_dr, get_dr

# Test 1: Kiểm tra tất cả DR có được register không
print("\n[TEST 1] Checking registration...")
expected_rules = ["EDD", "SPT", "LPT", "FCFS", "FIFO", "CR"]

for rule in expected_rules:
    if has_dr(rule):
        print(f"  ✓ {rule:6s} - Registered successfully")
    else:
        print(f"  ✗ {rule:6s} - NOT registered!")

# Test 2: Hiển thị tất cả DR trong registry
print("\n[TEST 2] All registered dispatching rules:")
print(f"  Total: {len(DR_REGISTRY)} rules")
for name in sorted(DR_REGISTRY.keys()):
    fn = DR_REGISTRY[name]
    print(f"  - {name:6s}: {fn.__name__}")

# Test 3: Verify functions có thể retrieve
print("\n[TEST 3] Retrieving functions...")
for rule in expected_rules:
    try:
        fn = get_dr(rule)
        print(f"  ✓ {rule:6s}: {fn}")
    except Exception as e:
        print(f"  ✗ {rule:6s}: Error - {e}")

# Test 4: Verify case-insensitive
print("\n[TEST 4] Testing case-insensitivity...")
test_cases = ["spt", "Spt", "SPT", "lpt", "LPT", "fcfs", "FCFS"]
for test_name in test_cases:
    try:
        fn = get_dr(test_name)
        print(f"  ✓ '{test_name}' → {fn.__name__}")
    except Exception as e:
        print(f"  ✗ '{test_name}' → Error: {e}")

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)
