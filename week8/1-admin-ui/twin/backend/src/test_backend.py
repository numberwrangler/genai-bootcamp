#!/usr/bin/env python3
"""
Simple test script to check backend functionality
"""
import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all imports work"""
    try:
        logger.info("Testing imports...")
        from app.main import app, question_manager
        logger.info("✓ All imports successful")
        return True
    except Exception as e:
        logger.error(f"✗ Import failed: {e}")
        return False

def test_environment():
    """Test environment variables"""
    logger.info("Testing environment variables...")
    
    required_vars = ["STATE_BUCKET", "DDB_TABLE"]
    missing_vars = []
    
    for var in required_vars:
        value = os.environ.get(var)
        if not value:
            missing_vars.append(var)
            logger.error(f"✗ Missing environment variable: {var}")
        else:
            logger.info(f"✓ {var}: {value}")
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        return False
    
    logger.info("✓ All environment variables set")
    return True

def test_question_manager():
    """Test question manager initialization"""
    try:
        logger.info("Testing question manager...")
        from app.questions import QuestionManager
        qm = QuestionManager()
        logger.info("✓ Question manager initialized successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Question manager failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("Starting backend tests...")
    
    tests = [
        test_imports,
        test_environment,
        test_question_manager
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        logger.info("")
    
    logger.info(f"Tests completed: {passed}/{total} passed")
    
    if passed == total:
        logger.info("✓ All tests passed! Backend should work correctly.")
        return 0
    else:
        logger.error("✗ Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
