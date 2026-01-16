try:
    import mediapipe as mp
    print("MediaPipe imported successfully")
    print(f"File: {mp.__file__}")
    print(f"Dir(mp): {dir(mp)}")
    
    try:
        import mediapipe.python.solutions as solutions
        print("Imported mediapipe.python.solutions directly")
        print(dir(solutions))
    except ImportError as e:
        print(f"Failed to import mediapipe.python.solutions: {e}")

except Exception as e:
    print(f"Error: {e}")
