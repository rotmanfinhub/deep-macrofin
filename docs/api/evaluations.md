# deep_macrofin.evaluations

## Formula

```py
class Formula(formula_str: str, evaluation_method: Union[EvaluationMethod, str], latex_var_mapping: Dict[str, str] = {})
```

Base class for string evaluations. Given a string representation of a formula, and a set of variables (state, value, prices, etc) in a model, parse the formula to a pytorch function that can be evaluated. 

Latex equation with restricted format is supported for initialization. When Latex is provided, it is first parsed into Python-evaluable strings with Latex-variable mappings provided by the user. For instance, to input the Cauchy-Euler equation: $x^2 y'' + 6xy' + 4y =0$, the user can either type in the raw Python string `x**2*y_xx + 6*x*y_x + 4*y = 0` or the Latex version `$x^2 * \frac{\partial^2 y}{\partial x^2} + 6*x*\frac{\partial y}{\partial x} + 4*y = 0$`. The Latex version will be internally parsed to the raw Python string version for evaluation.

**Parameters**:

- formula_str: **str**, the string version of the formula. If the provided formula_str is supposed to be a latex string, it must be $ enclosed and in the regular form, e.g. `formula_str=r"$x^2*y$"`, and all multiplication symbols must be explicitly provided as * in the equation.
- evaluation_method: **Union[EvaluationMethod, str]**, Enum, select from `eval`, `sympy`, `ast`, corresponding to the four methods below. For now, only eval is supported.
- latex_var_mapping: **Dict[str, str]**, only used if the formula_str is in latex form, the keys should be the latex expression, and the values should be the corresponding python variable name. All strings with single slash in latex must be defined as a raw string. All spaces in the key must match exactly as in the input formula_str. 

    **Example**:
    ```py
    latex_var_map = {
        r"\eta_t": "eta",
        r"\rho^i": "rhoi",
        r"\mu_t^{n h}": "munh",
        r"\sigma_t^{na}": "signa",
        r"\sigma_t^{n ia}": "signia",
        r"\sigma_t^{qa}": "sigqa",
        "c_t^i": "ci",
        "c_t^h": "ch",
    }
    ```

Evaluation Methods:
```py
class EvaluationMethod(str, Enum):
    Eval = "eval"
    Sympy = "sympy"
    AST = "ast"
```

### eval
```py
def eval(self, available_functions: Dict[str, Callable], variables: Dict[str, torch.Tensor]):
```

Evaluate the formula with existing functions and provided assignments to variables
This evaluates the function by simple string parsing.

**Parameters**:

- available_functions: **Dict[str, Callable]**, function calls attached to a string variable. It can include all `LearnableVar` and derivatives.
- variables: **Dict[str, torch.Tensor]**, values assigned to each variable.

## Comparator

A Enum class representing comparitor symbols, used in [Conditions](#baseconditions) and [Constraint](#constraint). 

```py
class Comparator(str, Enum):
    LEQ = "<="
    GEQ = ">="
    LT = "<"
    GT = ">"
    EQ = "="
```

## BaseConditions

```py
class BaseConditions(lhs: str, lhs_state: Dict[str, torch.Tensor], 
                    comparator: Comparator, 
                    rhs: str, rhs_state: Dict[str, torch.Tensor], 
                    label: str, latex_var_mapping: Dict[str, str] = {})
```

Define specific boundary/initial conditions for a specific agent. e.g. x(0)=0 or x(0)=x(1) (Periodic conditions). May also be an inequality, but it is very rare.

The difference between a constraint and a condition is:

- a constraint must be satisfied at any state
- a condition is satisfied at a specific given state

**Parameters**:

- lhs: **str**, the string expression for lhs formula, latex expression not supported, should be functions of specific format agent_name(SV), endog_name(SV), or simply a constant value
- lhs_state: **Dict[str, torch.Tensor]**, the specific value of SV to evaluate lhs at for the agent/endogenous variable
- comparator: **Comparator**
- rhs: **str**, the string expression for lhs formula, latex expression not supported, should be functions of specific format agent_name(SV), endog_name(SV), or simply a constant value
- rhs_state: **Dict[str, torch.Tensor]**, the specific value of SV to evaluate rhs at for the agent/endogenous variable, if rhs is a constant, this can be an empty dictionary
- label: **str**, label for the condition
- latex_var_mapping: **Not implemented**. only used if the formula_str is in latex form, the keys should be the latex expression, and the values should be the corresponding python variable name. 

**Example**:
```py
BaseConditions(lhs="f(SV)", lhs_state={"SV": torch.zeros((1,1))}, 
                comparator"=", 
                rhs"1", rhs_state={}, 
                label="eg1")
'''
The condition is f(0)=1
'''


BaseConditions(lhs="f(SV)", lhs_state={"SV": torch.zeros((1,1))}, 
                comparator"=", 
                rhs"f(SV)", rhs_state={"SV": torch.ones((1,1))}, 
                label="eg2")
'''
The condition is f(0)=f(1)
'''
```

### eval
```py
def eval(self, available_functions: Dict[str, Callable], 
                loss_reduction: LossReductionMethod=LossReductionMethod.MSE):
```

Computes the loss based on the required conditions:

$$\mathcal{L}_{cond} = \frac{1}{|U|} \sum_{x\in U}\|\mathcal{C}(v_i, x)\|_2^2$$

Details for evaluating non-equality conditions are the same as [Constraint](#constraint)


## AgentConditions

```py
class AgentConditions(agent_name: str, 
                    lhs: str, lhs_state: Dict[str, torch.Tensor], 
                    comparator: Comparator, 
                    rhs: str, rhs_state: Dict[str, torch.Tensor], 
                    label: str, latex_var_mapping: Dict[str, str] = {})
```

Subclass of `BaseConditions`. Defines conditions on agent with name `agent_name`.

## EndogVarConditions

```py
class EndogVarConditions(endog_name: str, 
                        lhs: str, lhs_state: Dict[str, torch.Tensor], 
                        comparator: Comparator, 
                        rhs: str, rhs_state: Dict[str, torch.Tensor], 
                        label: str, latex_var_mapping: Dict[str, str] = {})
```

Subclass of `BaseConditions`. Defines conditions on endogenous variable with name `endog_name`.

## Constraint

```py
class Constraint(lhs: str, comparator: Comparator, rhs: str, 
            label: str, latex_var_mapping: Dict[str, str] = {})
```

Given a string representation of a constraint (equality or inequality), and a set of variables (state, value, prices) in a model, parse the equation to a pytorch function that can be evaluated. If the constraint is an inequality, loss should be penalized whenever the inequality is not satisfied. 

### eval
```py
def eval(self, available_functions: Dict[str, Callable], variables: Dict[str, torch.Tensor], 
                loss_reduction: LossReductionMethod=LossReductionMethod.MSE):
```

**Example**: 
```py
Constraint(lhs="L", comparator=Comparator.GEQ, rhs="R", "eg1")
```
This represents the inequality: $L\geq R$, it is not satisfied when $R-L > 0$, so the loss is formulated as:

$$\mathcal{L}_{const} = \frac{1}{B} \|\text{ReLU}(R(x)-L(x))\|_2^2$$

If strict inequality is required,
```py
Constraint(lhs="L", comparator=Comparator.LT, rhs="R", "eg1")
```

This represents the inequality: $L< R$, it is not satisfied when $L-R \geq 0$, an additional $\epsilon=10^{-8}$ is added to the ReLU activation to ensure strictness.


## EndogEquation

```py
class EndogEquation(eq: str, label: str, latex_var_mapping: Dict[str, str] = {})
```

Given a string representation of an endogenuous equation, and a set of variables (state, value, prices) in a model. parse the LHS and RHS of the equation to pytorch functions that can be evaluated. This is used to define the algebraic (partial differential) equations as loss functions.

### eval
```py
def eval(self, available_functions: Dict[str, Callable], variables: Dict[str, torch.Tensor], 
                loss_reduction: LossReductionMethod=LossReductionMethod.MSE):
```

Computes the loss based on the equation:

$$\mathcal{L}_{endog} = \frac{1}{B} \|l(x)-r(x)\|_2^2$$

## Equation

```py
class Equation(eq: str, label: str, latex_var_mapping: Dict[str, str] = {})
```

Given a string representation of new variable definition, properly evaluate it with agent, endogenous variables, and constants. Assign new value to LHS.

### eval
```py
def eval(self, available_functions: Dict[str, Callable], variables: Dict[str, torch.Tensor]):
```

Compute and return the value of RHS, which will be assigned to LHS variable.

## HJBEquation

```py
class HJBEquation(eq: str, label: str, latex_var_mapping: Dict[str, str] = {})
```

Given a string representation of a Hamilton-Jacobi-Bellman equation, and a set of variables in a model, parse the equation to a pytorch function that can be evaluated.

### eval
```py
def eval(self, available_functions: Dict[str, Callable], variables: Dict[str, torch.Tensor], 
                loss_reduction: LossReductionMethod=LossReductionMethod.MSE):
```

Compute the MSE with zero as min/max problem.

## System
```py
class System(activation_constraints: List[Constraint], 
            label: str=None, 
            latex_var_mapping: Dict[str, str] = {})
```

Represents a system to be evaluated when `activation_constraints` are all satisfied.

### add_equation
```py
def add_equation(self, eq: str, label: str=None)
```
Add an equation to define a new variable within the system

### add_endog_equation
```py
def add_endog_equation(self, eq: str, label: str=None, weight=1.0, loss_reduction: LossReductionMethod=LossReductionMethod.MSE)
```
Add an equation for loss computation within the system

> Note: None reduction is not supported in a system

### add_constraint
```py
def add_constraint(self, lhs: str, comparator: Comparator, rhs: str, label: str=None, weight=1.0, loss_reduction: LossReductionMethod=LossReductionMethod.MSE)
```
Add an inequality constraint for loss computation within the system

> Note: None reduction is not supported in a system

### compute_constraint_mask
```py
def compute_constraint_mask(self, available_functions: Dict[str, Callable], variables: Dict[str, torch.Tensor])
```
Check if the constraint is satisfied. Need to check for each individual batch element. Get a mask $\mathbb{1}_{mask}$ for loss in each batch element.

### eval
```py
def eval(self, available_functions: Dict[str, Callable], variables: Dict[str, torch.Tensor])
```

Compute the loss based on the system constraint. Only elements in a batch that satisfy the `activation_constraints` are used in the loss computation.

$$\mathcal{L}_{endog, i} = \frac{1}{\sum \mathbb{1}_{mask}} \langle (l-r)^2, \mathbb{1}_{mask}\rangle$$

$$\mathcal{L}_{sys} = \sum_{i=1}^N \lambda_i \mathcal{L}_{endog, i}$$

## Loss Compute Methods

The constants determine which loss reduction method to use.

```py
class LossReductionMethod(str, Enum):
    MSE = "MSE" # mean squared error
    MAE = "MAE" # mean absolute error
    SSE = "SSE" # sum squared error
    SAE = "SAE" # sum absolute error
    NONE = "None" # no reduction

LOSS_REDUCTION_MAP = {
    LossReductionMethod.MSE: lambda x: torch.mean(torch.square(x)),
    LossReductionMethod.MAE: lambda x: torch.mean(torch.abs(x)),
    LossReductionMethod.SSE: lambda x: torch.sum(torch.square(x)),
    LossReductionMethod.SAE: lambda x: torch.sum(torch.absolute(x)),
    LossReductionMethod.NONE: lambda x: x,
}
```

