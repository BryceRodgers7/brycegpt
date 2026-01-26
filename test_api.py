"""
Test script for VoyagerGPT API
Run this to verify your deployment is working correctly
"""

import requests
import sys
import time


def test_api(base_url: str):
    """Test the VoyagerGPT API endpoints"""
    
    print("=" * 60)
    print("VoyagerGPT API Test Suite")
    print("=" * 60)
    print(f"Testing API at: {base_url}")
    print()
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Root endpoint
    print("Test 1: Root Endpoint")
    print("-" * 40)
    try:
        response = requests.get(f"{base_url}/", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status: {response.status_code}")
            print(f"   Message: {data.get('message')}")
            print(f"   Version: {data.get('version')}")
            tests_passed += 1
        else:
            print(f"❌ Failed with status: {response.status_code}")
            tests_failed += 1
    except Exception as e:
        print(f"❌ Error: {e}")
        tests_failed += 1
    print()
    
    # Test 2: Health check
    print("Test 2: Health Check")
    print("-" * 40)
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status: {response.status_code}")
            print(f"   API Status: {data.get('status')}")
            print(f"   Model Loaded: {data.get('model_loaded')}")
            print(f"   Device: {data.get('device')}")
            if data.get('status') == 'healthy' and data.get('model_loaded'):
                tests_passed += 1
            else:
                print("⚠️  API is responding but not healthy")
                tests_failed += 1
        else:
            print(f"❌ Failed with status: {response.status_code}")
            tests_failed += 1
    except Exception as e:
        print(f"❌ Error: {e}")
        tests_failed += 1
    print()
    
    # Test 3: Vocabulary endpoint
    print("Test 3: Vocabulary Endpoint")
    print("-" * 40)
    try:
        response = requests.get(f"{base_url}/vocab", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status: {response.status_code}")
            print(f"   Vocab Size: {data.get('vocab_size')}")
            print(f"   Block Size: {data.get('block_size')}")
            print(f"   Characters: {len(data.get('characters', []))} chars")
            tests_passed += 1
        else:
            print(f"❌ Failed with status: {response.status_code}")
            tests_failed += 1
    except Exception as e:
        print(f"❌ Error: {e}")
        tests_failed += 1
    print()
    
    # Test 4: Text generation (simple)
    print("Test 4: Text Generation (Simple)")
    print("-" * 40)
    try:
        payload = {
            "seed": 1337,
            "temperature": 0.1,
            "max_tokens": 50
        }
        print(f"   Payload: {payload}")
        print("   Generating... (this may take 10-30 seconds on CPU)")
        
        start_time = time.time()
        response = requests.post(
            f"{base_url}/generate",
            json=payload,
            timeout=60
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status: {response.status_code}")
            print(f"   Generation Time: {data.get('generation_time', 0):.2f}s")
            print(f"   Total Time: {elapsed:.2f}s")
            print(f"   Tokens Generated: {len(data.get('tokens', []))}")
            print(f"   Text Preview: {data.get('text', '')[:100]}...")
            tests_passed += 1
        else:
            print(f"❌ Failed with status: {response.status_code}")
            print(f"   Response: {response.text}")
            tests_failed += 1
    except requests.exceptions.Timeout:
        print("❌ Request timed out (generation took too long)")
        tests_failed += 1
    except Exception as e:
        print(f"❌ Error: {e}")
        tests_failed += 1
    print()
    
    # Test 5: Text generation with context
    print("Test 5: Text Generation (With Context)")
    print("-" * 40)
    try:
        # First generation
        payload1 = {
            "seed": 42,
            "temperature": 0.5,
            "max_tokens": 30
        }
        response1 = requests.post(
            f"{base_url}/generate",
            json=payload1,
            timeout=60
        )
        
        if response1.status_code == 200:
            tokens = response1.json()["tokens"]
            
            # Continue with context
            payload2 = {
                "seed": 42,
                "temperature": 0.5,
                "max_tokens": 30,
                "context": tokens
            }
            response2 = requests.post(
                f"{base_url}/generate",
                json=payload2,
                timeout=60
            )
            
            if response2.status_code == 200:
                data2 = response2.json()
                print(f"✅ Status: {response2.status_code}")
                print(f"   Context Length: {len(tokens)}")
                print(f"   New Tokens: {len(data2.get('tokens', [])) - len(tokens)}")
                print(f"   Total Tokens: {len(data2.get('tokens', []))}")
                tests_passed += 1
            else:
                print(f"❌ Second generation failed: {response2.status_code}")
                tests_failed += 1
        else:
            print(f"❌ First generation failed: {response1.status_code}")
            tests_failed += 1
    except Exception as e:
        print(f"❌ Error: {e}")
        tests_failed += 1
    print()
    
    # Test 6: Parameter validation
    print("Test 6: Parameter Validation")
    print("-" * 40)
    try:
        # Invalid temperature
        payload = {
            "seed": 1337,
            "temperature": 5.0,  # Too high
            "max_tokens": 50
        }
        response = requests.post(
            f"{base_url}/generate",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 422:  # Validation error expected
            print(f"✅ Validation working correctly (status: {response.status_code})")
            tests_passed += 1
        else:
            print(f"⚠️  Unexpected status: {response.status_code}")
            print("   (Expected 422 for invalid parameters)")
            tests_failed += 1
    except Exception as e:
        print(f"❌ Error: {e}")
        tests_failed += 1
    print()
    
    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"✅ Tests Passed: {tests_passed}")
    print(f"❌ Tests Failed: {tests_failed}")
    print(f"📊 Success Rate: {tests_passed / (tests_passed + tests_failed) * 100:.1f}%")
    print()
    
    if tests_failed == 0:
        print("🎉 All tests passed! Your API is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    # Get API URL from command line or use default
    if len(sys.argv) > 1:
        api_url = sys.argv[1]
    else:
        api_url = "http://localhost:8080"
    
    # Remove trailing slash if present
    api_url = api_url.rstrip('/')
    
    try:
        exit_code = test_api(api_url)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        sys.exit(1)

