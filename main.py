from fastapi import FastAPI
from pydantic import BaseModel, condecimal
from scipy.optimize import linprog
import numpy as np
import joblib


app = FastAPI()

# Load the trained linear regression model and budget optimization coefficients
model = joblib.load('linear_regression_model.pkl')
budget_coeffs = np.load('budget_optimization_coeffs.npy')
bounds = np.load('budget_bounds.npy')

# Extract individual channel bounds
min_tv, max_tv = bounds[0]
min_radio, max_radio = bounds[1]
min_newspaper, max_newspaper = bounds[2]

class BudgetRequest(BaseModel):
    budget: condecimal(gt=0)  # Ensures the budget is greater than 0

@app.post("/optimize-budget/")
def optimize_budget(request: BudgetRequest):
    
    # Define the objective function (negative means we maximize sales)
    c = -budget_coeffs 
    # Constraints: The total budget must be spent
    A = [[1, 1, 1]]
    b = [float(request.budget)]  # User-defined budget

    # Bounds: Min & max spend per channel
    bounds = [(min_tv, max_tv), (min_radio, max_radio), (min_newspaper, max_newspaper)]
    
    # Solve the optimization problem
    result = linprog(c, A_eq=A, b_eq=b, bounds=bounds, method="highs")

    if result.success:
        optimized_budget = result.x
        tv_spend, radio_spend, newspaper_spend = optimized_budget
        
        # Predict sales using the optimized budget
        user_data = np.array([[tv_spend, radio_spend, newspaper_spend]])
        predicted_sales = model.predict(user_data)[0]
        return {
            "Optimized TV Budget": round(tv_spend, 2),
            "Optimized Radio Budget": round(radio_spend, 2),
            "Optimized Newspaper Budget": round(newspaper_spend, 2),
            "Predicted Sales": round(predicted_sales, 2)
        }
    else:
        return {"error": "Optimization failed. Try a different budget."}
