import traceback
from src.api.model_loader import load_model
from src.api.predict import run_inference

if __name__ == '__main__':
    try:
        model = load_model()
        print('Model Loaded')
        result = run_inference(model)
        print("Inference Result:", result)
    except Exception as e:
        traceback.print_exc()
