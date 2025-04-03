from fastapi import FastAPI
from pydantic import BaseModel
from scipy.optimize import linprog
import numpy as np

app = FastAPI()

class BudgetRequest(BaseModel):
    budget: float  # Total budget

@app.post("/optimize-budget/")
def optimize_budget(request: BudgetRequest):
    # Define the objective function (negative means we maximize sales)
    c = np.array([-200, -150, -100])  # Example coefficients (change based on real data)

    # Constraints: The total budget must be spent
    A = [[1, 1, 1]]
    b = [request.budget]  # User-defined budget

    # Bounds: Min & max spend per channel
    bounds = [(1000, 30000), (500, 15000), (200, 10000)]

    # Solve the optimization problem
    result = linprog(c, A_eq=A, b_eq=b, bounds=bounds, method="highs")

    if result.success:
        return {
            "TV": round(result.x[0], 2),
            "Radio": round(result.x[1], 2),
            "Newspaper": round(result.x[2], 2),
        }
    else:
        return {"error": "Optimization failed. Try a different budget."}
