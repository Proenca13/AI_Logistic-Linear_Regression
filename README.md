# Implementing Regularized Logistic and Linear Regression with a Base Model

## Overview
This project focuses on implementing **Linear Regression** and **Logistic Regression** from scratch, incorporating **L2 regularization (Ridge)** to prevent overfitting and improve generalization. Additionally, a **Base Model** is developed to provide a reusable structure for building machine learning models, ensuring modularity, scalability, and maintainability.

## Objectives
1. **Implement a Base Model class** that defines essential methods such as:
   - `fit()`
   - `predict()`
   - `score()`
   - Support for common utilities like cost calculation and gradient updates.
   
2. **Implement Linear Regression** with:
   - Mean Squared Error (MSE) as the cost function.
   - **L2 (Ridge) regularization**.

3. **Implement Logistic Regression** with:
   - Binary cross-entropy (log loss) as the cost function.
   - **L2 (Ridge) regularization**.

4. **Ensure flexibility** to:
   - Adjust learning rate and number of iterations.
   - Enable or disable regularization.
   - Work with various datasets.

5. **Evaluate models** on synthetic and/or real datasets:
   - For **Linear Regression**: Regression metrics like RMSE, MAE, or R² score.
   - For **Logistic Regression**: Classification metrics like Accuracy, Precision, Recall, and F1-score.

## Technical Requirements
- Implementations must be **from scratch** using **Python** (without using machine learning libraries such as scikit-learn for the core algorithm).  
- Use **NumPy** for numerical computations.  
- Provide **clear and modular code** with proper documentation and comments.  

## Project Structure

├── base_model.py # Abstract Base Model class
├── linear_regression.py # Implementation of Linear Regression with L2 regularization
├── logistic_regression.py # Implementation of Logistic Regression with L2 regularization
├── utils.py # Helper functions (cost functions, metrics, etc.)
├── tests/ # Unit tests for each component
├── examples/ # Example notebooks or scripts demonstrating usage
└── README.md # Project documentation
