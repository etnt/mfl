"""
Hindley-Milner Type Inference System Implementation
"""

import dataclasses
from typing import Dict, Any, Optional
from mfl_ast import (
    ASTNode, Var, Int, Bool, Function, Apply, Let, If, BinOp, UnaryOp
)

@dataclasses.dataclass
class MonoType:
    """Base class for monomorphic types (types without quantifiers)."""
    def find(self) -> 'MonoType':
        """Find the ultimate type that this type resolves to."""
        return self

    def __str__(self):
        return self.__repr__()

@dataclasses.dataclass
class TyVar(MonoType):
    """Represents a type variable that can be unified with any type."""
    name: str
    forwarded: MonoType = None

    def find(self) -> 'MonoType':
        """Follow the chain of forwarded references to find the ultimate type."""
        result = self
        while isinstance(result, TyVar) and result.forwarded:
            result = result.forwarded
        return result

    def make_equal_to(self, other: MonoType):
        """Unify this type variable with another type."""
        chain_end = self.find()
        assert isinstance(chain_end, TyVar)
        chain_end.forwarded = other

    def __repr__(self):
        if self.forwarded:
            return str(self.find())
        return self.name

@dataclasses.dataclass
class TyCon(MonoType):
    """Represents a type constructor."""
    name: str
    args: list

    def __repr__(self):
        if not self.args:
            return self.name
        args_str = ", ".join(str(arg) for arg in self.args)
        return f"{self.name}({args_str})"

@dataclasses.dataclass
class Forall:
    """Represents a polymorphic type scheme (∀a.T)."""
    vars: list
    ty: MonoType

    def __repr__(self):
        if not self.vars:
            return str(self.ty)
        vars_str = " ".join(self.vars)
        return f"∀{vars_str}.{self.ty}"

def occurs_in(var: TyVar, ty: MonoType) -> bool:
    """Check if a type variable occurs within a type."""
    ty = ty.find()
    if ty == var:
        return True
    if isinstance(ty, TyCon):
        return any(occurs_in(var, arg) for arg in ty.args)
    return False

def unify_j(ty1: MonoType, ty2: MonoType):
    """Unify two types, making them equal by finding appropriate substitutions."""
    ty1 = ty1.find()
    ty2 = ty2.find()
    if isinstance(ty1, TyVar):
        if occurs_in(ty1, ty2):
            raise Exception(f"Recursive type found: {ty1} and {ty2}")
        ty1.make_equal_to(ty2)
        return
    if isinstance(ty2, TyVar):
        return unify_j(ty2, ty1)
    if isinstance(ty1, TyCon) and isinstance(ty2, TyCon):
        if ty1.name != ty2.name or len(ty1.args) != len(ty2.args):
            raise Exception(f"Type mismatch: {ty1} and {ty2}")
        for l, r in zip(ty1.args, ty2.args):
            unify_j(l, r)

def update_node_type(node: Any, type: MonoType):
    """Helper function to update a node's type and recursively update child nodes"""
    if not isinstance(node, ASTNode):
        return
    
    resolved_type = type.find()
    node.type = resolved_type
    
    if isinstance(node, Function):
        # For functions, the type is -> (arg_type, return_type)
        if isinstance(resolved_type, TyCon) and resolved_type.name == "->" and len(resolved_type.args) == 2:
            arg_type, return_type = resolved_type.args
            node.arg.type = arg_type
            node.body.type = return_type
            update_node_type(node.body, return_type)
    
    elif isinstance(node, Apply):
        # For application, func should have type arg -> result
        func_type = TyCon("->", [node.arg.type, resolved_type])
        if isinstance(node.func, Var):
            node.func.type = func_type
        else:
            update_node_type(node.func, func_type)
        update_node_type(node.arg, node.arg.type)
    
    elif isinstance(node, Let):
        # For let expressions, propagate types to all components
        node.name.type = node.value.type
        update_node_type(node.value, node.value.type)
        update_node_type(node.body, resolved_type)
    
    elif isinstance(node, If):
        # For if expressions, condition must be boolean, both branches must have same type
        node.cond.type = BoolType
        node.then_expr.type = resolved_type
        node.else_expr.type = resolved_type
        update_node_type(node.cond, BoolType)
        update_node_type(node.then_expr, resolved_type)
        update_node_type(node.else_expr, resolved_type)
    
    elif isinstance(node, BinOp):
        # For binary operations, set types based on the operation
        if node.op in ["+", "-", "*", "/"]:
            node.left.type = IntType
            node.right.type = IntType
            update_node_type(node.left, IntType)
            update_node_type(node.right, IntType)
        elif node.op in ["&", "|"]:
            node.left.type = BoolType
            node.right.type = BoolType
            update_node_type(node.left, BoolType)
            update_node_type(node.right, BoolType)
        elif node.op in [">", "<", "==", "<=", ">="]:
            node.left.type = IntType
            node.right.type = IntType
            update_node_type(node.left, IntType)
            update_node_type(node.right, IntType)
    
    elif isinstance(node, UnaryOp):
        # For unary operations
        if node.op == "!":
            node.operand.type = BoolType
            update_node_type(node.operand, BoolType)
    
    elif isinstance(node, Int):
        node.type = IntType
    
    elif isinstance(node, Bool):
        node.type = BoolType

def infer_j(expr, ctx: Dict[str, Forall]) -> MonoType:
    """Infer the type of an expression in a given typing context."""
    result = fresh_tyvar()

    if isinstance(expr, Var):
        # Variables get their type from the context
        scheme = ctx.get(expr.name)
        if scheme is None:
            raise Exception(f"Unbound variable {expr.name}")
        unify_j(result, scheme.ty)
        expr.type = scheme.ty.find()

    elif isinstance(expr, Int):
        # Integer literals always have type 'int'
        unify_j(result, IntType)
        expr.type = IntType

    elif isinstance(expr, Bool):
        # Boolean literals always have type 'bool'
        unify_j(result, BoolType)
        expr.type = BoolType

    elif isinstance(expr, If):
        # For if expressions:
        # 1. Condition must be boolean
        cond_type = infer_j(expr.cond, ctx)
        unify_j(cond_type, BoolType)
        
        # 2. Infer types for both branches
        then_type = infer_j(expr.then_expr, ctx)
        else_type = infer_j(expr.else_expr, ctx)
        
        # 3. Both branches must have the same type
        unify_j(then_type, else_type)
        
        # 4. Result type is the same as branch type
        unify_j(result, then_type)
        
        # Update types
        expr.type = then_type.find()
        expr.cond.type = BoolType
        expr.then_expr.type = expr.type
        expr.else_expr.type = expr.type

    elif isinstance(expr, Function):
        # For a function λx.e:
        # 1. Create fresh type variable for the argument
        arg_type = fresh_tyvar()
        # Update the argument's type
        expr.arg.type = arg_type
        # 2. Add x:arg_type to context and infer the body
        body_ctx = ctx.copy()
        body_ctx[expr.arg.name] = Forall([], arg_type)
        body_type = infer_j(expr.body, body_ctx)
        # 3. Create function type arg_type → body_type
        func_type = TyCon("->", [arg_type, body_type])
        unify_j(result, func_type)

    elif isinstance(expr, Apply):
        # For function application (f x):
        # 1. Infer types for function and argument
        func_type = infer_j(expr.func, ctx)
        arg_type = infer_j(expr.arg, ctx)
        # 2. Create fresh type variable for the result
        ret_type = fresh_tyvar()
        # 3. Unify func_type with arg_type → ret_type
        expected_func_type = TyCon("->", [arg_type, ret_type])
        unify_j(func_type, expected_func_type)
        unify_j(result, ret_type)
        # 4. Update the function's type to show it's a function type
        if isinstance(expr.func, Var):
            expr.func.type = expected_func_type

    elif isinstance(expr, Let):
        # For let x = e1 in e2:
        # 1. Infer type of e1
        value_type = infer_j(expr.value, ctx)
        # Update the name's type
        expr.name.type = value_type
        # 2. Extend context with x:value_type and infer e2
        body_ctx = ctx.copy()
        body_ctx[expr.name.name] = Forall([], value_type)
        body_type = infer_j(expr.body, body_ctx)
        unify_j(result, body_type)

    elif isinstance(expr, BinOp):
        # For binary operations:
        # 1. Infer types of both operands
        left_type = infer_j(expr.left, ctx)
        right_type = infer_j(expr.right, ctx)

        # 2. Handle different types of operations
        if expr.op in ["+", "-", "*", "/"]:
            # Arithmetic operations require integers
            unify_j(left_type, IntType)
            unify_j(right_type, IntType)
            unify_j(result, IntType)
            expr.type = IntType
            expr.left.type = IntType
            expr.right.type = IntType
        elif expr.op in ["&", "|"]:
            # Boolean operations require booleans
            unify_j(left_type, BoolType)
            unify_j(right_type, BoolType)
            unify_j(result, BoolType)
            expr.type = BoolType
            expr.left.type = BoolType
            expr.right.type = BoolType
        elif expr.op in [">", "<", "==", "<=", ">="]:
            # Comparison operations require integers and return boolean
            unify_j(left_type, IntType)
            unify_j(right_type, IntType)
            unify_j(result, BoolType)
            expr.type = BoolType
            expr.left.type = IntType
            expr.right.type = IntType
        else:
            raise Exception(f"Unknown binary operator: {expr.op}")

    elif isinstance(expr, UnaryOp):
        # For unary operations (currently only boolean not):
        # 1. Infer type of operand
        operand_type = infer_j(expr.operand, ctx)

        # 2. Handle not operation
        if expr.op == "!":
            # Not operation requires boolean operand
            unify_j(operand_type, BoolType)
            unify_j(result, BoolType)
            expr.type = BoolType
            expr.operand.type = BoolType
        else:
            raise Exception(f"Unknown unary operator: {expr.op}")

    else:
        raise Exception(f"Unknown expression type: {type(expr)}")

    # Store the inferred type in the AST node and update child nodes
    final_type = result.find()
    expr.type = final_type
    update_node_type(expr, final_type)
    return final_type

# Type Variable Generation
tyvar_counter = 0

def fresh_tyvar(prefix="a"):
    """Generate a fresh type variable with a unique name."""
    global tyvar_counter
    tyvar_counter += 1
    return TyVar(name=f"{prefix}{tyvar_counter}")

# Primitive Types
IntType = TyCon("int", [])    # The integer type
BoolType = TyCon("bool", [])  # The boolean type
