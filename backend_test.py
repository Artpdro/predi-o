import requests
import sys
import json
from datetime import datetime

class RouteSafetyAPITester:
    def __init__(self, base_url="https://xgroute-app.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0

    def run_test(self, name, method, endpoint, expected_status, data=None, timeout=30):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}" if not endpoint.startswith('http') else endpoint
        headers = {'Content-Type': 'application/json'}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=timeout)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    print(f"   Response keys: {list(response_data.keys()) if isinstance(response_data, dict) else 'Non-dict response'}")
                    return True, response_data
                except:
                    return True, response.text
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                print(f"   Response: {response.text[:200]}...")
                return False, {}

        except requests.exceptions.Timeout:
            print(f"❌ Failed - Request timeout after {timeout}s")
            return False, {}
        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_root_endpoint(self):
        """Test the root API endpoint"""
        success, response = self.run_test(
            "Root API Endpoint",
            "GET",
            "",
            200
        )
        if success and isinstance(response, dict):
            print(f"   Model ready: {response.get('model_ready', 'Unknown')}")
        return success

    def test_analyze_route_basic(self):
        """Test basic route analysis"""
        test_data = {
            "start_address": "São Paulo, SP",
            "end_address": "Rio de Janeiro, RJ"
        }
        
        success, response = self.run_test(
            "Basic Route Analysis",
            "POST",
            "analyze-route",
            200,
            data=test_data,
            timeout=60  # Longer timeout for ML processing
        )
        
        if success and isinstance(response, dict):
            required_fields = ['id', 'start_address', 'end_address', 'overall_risk_score', 
                             'safety_recommendation', 'total_distance_km', 'estimated_accidents',
                             'route_points', 'risk_factors']
            
            missing_fields = [field for field in required_fields if field not in response]
            if missing_fields:
                print(f"   ⚠️  Missing fields: {missing_fields}")
            else:
                print(f"   ✅ All required fields present")
                print(f"   Risk Score: {response.get('overall_risk_score', 'N/A')}")
                print(f"   Distance: {response.get('total_distance_km', 'N/A')} km")
                print(f"   Route Points: {len(response.get('route_points', []))}")
        
        return success, response

    def test_analyze_route_different_cities(self):
        """Test route analysis with different Brazilian cities"""
        test_cases = [
            {"start_address": "Brasília, DF", "end_address": "Salvador, BA"},
            {"start_address": "Fortaleza, CE", "end_address": "Belo Horizonte, MG"},
            {"start_address": "Manaus, AM", "end_address": "Curitiba, PR"}
        ]
        
        all_success = True
        for i, test_data in enumerate(test_cases):
            success, response = self.run_test(
                f"Route Analysis {i+1} ({test_data['start_address']} → {test_data['end_address']})",
                "POST",
                "analyze-route",
                200,
                data=test_data,
                timeout=60
            )
            if success and isinstance(response, dict):
                print(f"   Risk Score: {response.get('overall_risk_score', 'N/A')}")
            all_success = all_success and success
        
        return all_success

    def test_analyze_route_invalid_input(self):
        """Test route analysis with invalid input"""
        test_cases = [
            {"start_address": "", "end_address": "Rio de Janeiro, RJ"},
            {"start_address": "São Paulo, SP", "end_address": ""},
            {"start_address": "Invalid City", "end_address": "Another Invalid City"}
        ]
        
        for i, test_data in enumerate(test_cases):
            success, response = self.run_test(
                f"Invalid Input Test {i+1}",
                "POST",
                "analyze-route",
                200,  # Should still work with mock geocoding
                data=test_data,
                timeout=30
            )
            # Even invalid cities should work due to mock geocoding fallback

    def test_accident_data_endpoint(self):
        """Test accident data endpoint for heatmap"""
        success, response = self.run_test(
            "Accident Data Endpoint",
            "GET",
            "accident-data",
            200,
            timeout=30
        )
        
        if success and isinstance(response, dict):
            accidents = response.get('accidents', [])
            print(f"   Accident records: {len(accidents)}")
            if accidents:
                sample = accidents[0]
                required_fields = ['lat', 'lng', 'severity', 'date', 'cause', 'type']
                missing_fields = [field for field in required_fields if field not in sample]
                if missing_fields:
                    print(f"   ⚠️  Missing fields in accident data: {missing_fields}")
                else:
                    print(f"   ✅ Accident data structure correct")
                    print(f"   Sample: lat={sample.get('lat')}, lng={sample.get('lng')}, severity={sample.get('severity')}")
        
        return success

    def test_route_history_endpoint(self):
        """Test route history endpoint"""
        success, response = self.run_test(
            "Route History Endpoint",
            "GET",
            "route-history",
            200
        )
        
        if success:
            if isinstance(response, list):
                print(f"   Route history records: {len(response)}")
                if response:
                    sample = response[0]
                    print(f"   Sample route: {sample.get('start_address', 'N/A')} → {sample.get('end_address', 'N/A')}")
            else:
                print(f"   ⚠️  Expected list, got {type(response)}")
        
        return success

    def test_data_persistence(self):
        """Test that route analyses are stored and retrievable"""
        print(f"\n🔍 Testing Data Persistence...")
        
        # First, analyze a route
        test_data = {
            "start_address": "Test City A",
            "end_address": "Test City B"
        }
        
        success1, analysis_response = self.run_test(
            "Route Analysis for Persistence Test",
            "POST",
            "analyze-route",
            200,
            data=test_data,
            timeout=60
        )
        
        if not success1:
            return False
            
        # Then check if it appears in history
        success2, history_response = self.run_test(
            "Route History After Analysis",
            "GET",
            "route-history",
            200
        )
        
        if success2 and isinstance(history_response, list):
            # Check if our test route is in the history
            found = any(
                route.get('start_address') == 'Test City A' and 
                route.get('end_address') == 'Test City B'
                for route in history_response
            )
            if found:
                print(f"   ✅ Route analysis successfully stored and retrieved")
                return True
            else:
                print(f"   ⚠️  Route analysis not found in history")
                return False
        
        return False

def main():
    print("🚀 Starting Route Safety Predictor API Tests")
    print("=" * 60)
    
    tester = RouteSafetyAPITester()
    
    # Test sequence
    tests = [
        ("Root Endpoint", tester.test_root_endpoint),
        ("Accident Data", tester.test_accident_data_endpoint),
        ("Route History", tester.test_route_history_endpoint),
        ("Basic Route Analysis", tester.test_analyze_route_basic),
        ("Different Cities", tester.test_analyze_route_different_cities),
        ("Invalid Input Handling", tester.test_analyze_route_invalid_input),
        ("Data Persistence", tester.test_data_persistence),
    ]
    
    print(f"\nRunning {len(tests)} test suites...")
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            test_func()
        except Exception as e:
            print(f"❌ Test suite failed with exception: {str(e)}")
    
    # Print final results
    print(f"\n{'='*60}")
    print(f"📊 FINAL RESULTS")
    print(f"Tests passed: {tester.tests_passed}/{tester.tests_run}")
    print(f"Success rate: {(tester.tests_passed/tester.tests_run*100):.1f}%" if tester.tests_run > 0 else "No tests run")
    
    if tester.tests_passed == tester.tests_run:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed - check logs above")
        return 1

if __name__ == "__main__":
    sys.exit(main())