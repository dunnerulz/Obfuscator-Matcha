"""
Matcha LuaU Obfuscator
======================
A performance-friendly obfuscator designed specifically for the Matcha LuaU VM.
Focuses on making deobfuscation extremely difficult while maintaining runtime efficiency.

Key Design Principles:
- Constant Pool: All strings and numbers are extracted and stored in a hidden table
- Compatible with Matcha's limitations (no .Magnitude, .Unit, no task.defer/delay/cancel)
- Optimized for high FPS execution (up to 1000fps with wait(.001))
"""

import random
import string
import re
from luaparser import ast
from luaparser import astnodes
from luaparser.astnodes import *
from luaparser.astnodes import Index, Name
# Explicitly import nodes we might use directly or need for dynamic resolution
# REMOVED 'True' from this import list because it is a Python keyword and causes a SyntaxError
# ADDED LocalFunction, Function for hoisting logic
from luaparser.astnodes import If, Block, While, Break, Invoke, Call, Number, Assign, LocalAssign, LocalFunction, Function, Return, Repeat, Do

# -------------------------------------------------------------------------
# DYNAMIC NODE DISCOVERY (AUTO-DETECT)
# -------------------------------------------------------------------------
SPECIFIC_BINOP_MODE = False # Flag to track if we need BinOp(Op(), L, R) or Op(L, R)

def get_statements(tree):
    """Helper to safely extract the list of statements from a parsed tree/chunk."""
    # Chunk -> Block -> body (list)
    if hasattr(tree, 'body'):
        if isinstance(tree.body, list):
            return tree.body
        if hasattr(tree.body, 'body') and isinstance(tree.body.body, list):
            return tree.body.body
    return []

def discover_node_classes():
    """Parses sample code to find the correct node classes for this environment."""
    global BINOP, ADD_OP, SUB_OP, MULT_OP, DIV_OP, EQ_OP, TRUE_NODE
    global NUMERIC_FOR, GENERIC_FOR, SPECIFIC_BINOP_MODE
    global ANON_FUNC
    global BXOR_OP, BAND_OP, BNOT_OP  # Bitwise operators for MBA
    
    # Initialize bitwise operators
    BXOR_OP = None
    BAND_OP = None
    BNOT_OP = None
    
    # 1. Discover Binary Operation Structure
    try:
        # Parse Addition
        add_tree = ast.parse("local x = 1 + 1")
        add_stmts = get_statements(add_tree)
        add_node = add_stmts[0].values[0] if add_stmts else None

        # Parse Equality
        eq_tree = ast.parse("local x = 1 == 1")
        eq_stmts = get_statements(eq_tree)
        eq_node = eq_stmts[0].values[0] if eq_stmts else None

        if add_node and hasattr(add_node, 'op'):
            # Generic Mode: BinOp(op=Add(), left=..., right=...)
            SPECIFIC_BINOP_MODE = False
            BINOP = type(add_node)
            ADD_OP = type(add_node.op)
            
            # Need to re-fetch sub/mult/div classes by parsing
            sub_tree = ast.parse("local x = 1 - 1")
            SUB_OP = type(get_statements(sub_tree)[0].values[0].op)
            
            mult_tree = ast.parse("local x = 1 * 1")
            MULT_OP = type(get_statements(mult_tree)[0].values[0].op)
            
            div_tree = ast.parse("local x = 1 / 1")
            DIV_OP = type(get_statements(div_tree)[0].values[0].op)
            
            if eq_node:
                EQ_OP = type(eq_node.op)
            else:
                EQ_OP = getattr(astnodes, "Eq", getattr(astnodes, "EQ", None))
            
            # Discover bitwise operators for MBA
            try:
                bxor_tree = ast.parse("local x = bit32.bxor(1, 1)")
                band_tree = ast.parse("local x = bit32.band(1, 1)")
                bnot_tree = ast.parse("local x = bit32.bnot(1)")
                # These are function calls, not operators - we'll handle them differently
            except:
                pass

        elif add_node:
            # Specific Mode: AddOp(left=..., right=...)
            # The node itself IS the operator class
            SPECIFIC_BINOP_MODE = True
            BINOP = None # Not used in this mode
            ADD_OP = type(add_node)
            
            sub_tree = ast.parse("local x = 1 - 1")
            SUB_OP = type(get_statements(sub_tree)[0].values[0])
            
            mult_tree = ast.parse("local x = 1 * 1")
            MULT_OP = type(get_statements(mult_tree)[0].values[0])
            
            div_tree = ast.parse("local x = 1 / 1")
            DIV_OP = type(get_statements(div_tree)[0].values[0])
            
            if eq_node:
                EQ_OP = type(eq_node)
            else:
                EQ_OP = getattr(astnodes, "Eq", getattr(astnodes, "EQ", None))
        else:
            # Fallback (Should not happen if parsing works)
            SPECIFIC_BINOP_MODE = False
            BINOP = getattr(astnodes, "BinOp", None)
            ADD_OP = getattr(astnodes, "Add", None)
            EQ_OP = getattr(astnodes, "Eq", None)

    except Exception as e:
        print(f"[-] Discovery Error (BinOp): {e}")
        # Panic Fallback
        SPECIFIC_BINOP_MODE = False
        BINOP = getattr(astnodes, "BinOp", None)
        ADD_OP = getattr(astnodes, "Add", None)
        EQ_OP = getattr(astnodes, "Eq", None)
    
    # 3. Discover True Node
    try:
        tree = ast.parse("while true do end")
        stmts = get_statements(tree)
        if stmts:
            TRUE_NODE = type(stmts[0].condition)
        else:
            TRUE_NODE = None
    except:
        TRUE_NODE = None

    # 4. Discover Loop Nodes
    try:
        tree = ast.parse("for i=1,10 do end")
        stmts = get_statements(tree)
        NUMERIC_FOR = type(stmts[0]) if stmts else getattr(astnodes, "NumericFor", None)
    except:
        NUMERIC_FOR = getattr(astnodes, "NumericFor", None)

    try:
        tree = ast.parse("for i,v in pairs({}) do end")
        stmts = get_statements(tree)
        GENERIC_FOR = type(stmts[0]) if stmts else getattr(astnodes, "GenericFor", None)
    except:
        GENERIC_FOR = getattr(astnodes, "GenericFor", None)

    # 5. Discover Anonymous Function Node
    try:
        tree = ast.parse("local x = function() end")
        stmts = get_statements(tree)
        # stmts[0] is LocalAssign, values[0] is AnonymousFunction
        ANON_FUNC = type(stmts[0].values[0])
    except Exception as e:
        print(f"[-] Discovery Error (AnonFunc): {e}")
        ANON_FUNC = getattr(astnodes, "AnonymousFunction", None)

    # Debug Output
    mode_str = "Specific (OpNode)" if SPECIFIC_BINOP_MODE else "Generic (BinOp + Op)"
    print(f"[+] AST Mode: {mode_str}")
    print(f"[+] Auto-Discovered: Add={ADD_OP.__name__ if ADD_OP else '?'}, Eq={EQ_OP.__name__ if EQ_OP else '?'}, Anon={ANON_FUNC.__name__ if ANON_FUNC else '?'}")

# Execute Discovery
discover_node_classes()

class MatchaObfuscator:
    """
    Main obfuscator class that handles the transformation of Lua source code.
    
    Attributes:
        input_file (str): Path to the input .lua file to obfuscate
        output_file (str): Path where the obfuscated output will be written
        constants (list): Constant Pool - stores all extracted strings and numbers
        ast: The parsed Abstract Syntax Tree of the source code
    """
    
    def __init__(self, input_file: str, output_file: str):
        """
        Initialize the obfuscator with input/output file paths.
        
        Args:
            input_file: Path to the source .lua file
            output_file: Path for the obfuscated output file
        """
        # Variable Renaming (must be first so generate_var_name is available)
        self.var_count = 0
        self.var_prefix = "_v"
        self.used_names = set()  # Track used names to prevent collisions
        self.id_charset = "Il1O0"  # Confusing charset for variable names
        
        self.input_file = input_file
        self.output_file = output_file
        
        # Encryption Key for Strings
        # A random byte value (1-255) used for XOR encryption of string constants
        self.xor_key = random.randint(1, 255)
        
        # Constant Pool: All strings and numbers will be extracted here
        # This makes static analysis harder as values are no longer inline
        self.constants = []
        self.pool_name = self.generate_var_name()  # Randomized constant table name

        # List of sensitive globals to protect
        self.api_globals = ["Drawing", "Vector3", "Color3", "Instance", "CFrame", "game", "workspace", "wait", "spawn", "iskeypressed", "mouse1click", "WorldToScreen"]
        
        # AST will be populated by load_source()
        self.ast = None

    def generate_var_name(self):
        """
        Generates a unique obfuscated variable name using confusing characters.
        
        Uses characters that look similar (I, l, 1, O, 0) to make reverse
        engineering more difficult. Ensures the name doesn't start with a
        digit (invalid in Lua) and is unique across all generated names.
        
        Returns:
            str: A unique obfuscated variable name (e.g., IlI1O0lI).
        """
        # Characters that can start a variable name (no digits)
        first_char_set = "IlO"
        
        while True:
            # Generate random length between 8 and 12
            length = random.randint(8, 12)
            
            # First character must not be a digit
            first_char = random.choice(first_char_set)
            
            # Rest of the name can use the full confusing charset
            rest = ''.join(random.choice(self.id_charset) for _ in range(length - 1))
            
            name = first_char + rest
            
            # Ensure uniqueness
            if name not in self.used_names:
                self.used_names.add(name)
                self.var_count += 1  # Keep count for stats
                return name

    def generate_mutated_expression(self, value, depth=0):
        """
        Turns a simple integer into an obfuscated math expression.
        
        Uses only addition of positive numbers to guarantee correct results.
        The expressions look complex but always evaluate to the correct value.
        
        Args:
            value (int): The target number (must be positive for array indices).
            depth (int): Recursion depth to prevent infinite nesting.
            
        Returns:
            Node: An AST node representing the math expression.
        """
        # Ensure we have required operators
        if not isinstance(value, (int, float)) or not ADD_OP or not SUB_OP:
            return Number(value)
        
        # Convert to int
        value = int(value)
        
        # For very small or negative values, just return the raw number
        if value <= 1:
            return Number(value)
        
        # Limit recursion depth
        max_depth = 1  # Keep it shallow to avoid issues
        
        if depth >= max_depth:
            # At max depth, just split into two positive numbers
            a = random.randint(1, value - 1)
            b = value - a
            if SPECIFIC_BINOP_MODE:
                return ADD_OP(Number(a), Number(b))
            else:
                return BINOP(ADD_OP(), Number(a), Number(b))
        
        # Choose mutation strategy
        strategy = random.randint(1, 4)
        
        if strategy == 1:
            # Simple split: value = a + b
            a = random.randint(1, value - 1)
            b = value - a
            a_node = self.generate_mutated_expression(a, depth + 1)
            b_node = self.generate_mutated_expression(b, depth + 1)
            if SPECIFIC_BINOP_MODE:
                return ADD_OP(a_node, b_node)
            else:
                return BINOP(ADD_OP(), a_node, b_node)
        
        elif strategy == 2:
            # Triple split: value = a + b + c
            a = random.randint(1, max(1, value // 3))
            remainder = value - a
            b = random.randint(1, max(1, remainder - 1))
            c = remainder - b
            if c < 1:
                c = 1
                b = remainder - 1
            if b < 1:
                # Fallback to simple split
                a = random.randint(1, value - 1)
                b = value - a
                if SPECIFIC_BINOP_MODE:
                    return ADD_OP(Number(a), Number(b))
                else:
                    return BINOP(ADD_OP(), Number(a), Number(b))
            
            if SPECIFIC_BINOP_MODE:
                ab_node = ADD_OP(Number(a), Number(b))
                return ADD_OP(ab_node, Number(c))
            else:
                ab_node = BINOP(ADD_OP(), Number(a), Number(b))
                return BINOP(ADD_OP(), ab_node, Number(c))
        
        elif strategy == 3:
            # Subtraction with larger first operand: value = (value + offset) - offset
            offset = random.randint(50, 300)
            larger = value + offset
            if SPECIFIC_BINOP_MODE:
                return SUB_OP(Number(larger), Number(offset))
            else:
                return BINOP(SUB_OP(), Number(larger), Number(offset))
        
        else:
            # Multi-term: value = a + b + c - d where (a + b + c) > d and result = value
            # Keep it simple: a + b + c = value + d
            d = random.randint(50, 200)
            total = value + d
            a = random.randint(1, max(1, total // 3))
            remainder = total - a
            b = random.randint(1, max(1, remainder // 2))
            c = remainder - b
            
            if a < 1 or b < 1 or c < 1:
                # Fallback
                if SPECIFIC_BINOP_MODE:
                    return ADD_OP(Number(value - 1), Number(1))
                else:
                    return BINOP(ADD_OP(), Number(value - 1), Number(1))
            
            if SPECIFIC_BINOP_MODE:
                ab_node = ADD_OP(Number(a), Number(b))
                abc_node = ADD_OP(ab_node, Number(c))
                return SUB_OP(abc_node, Number(d))
            else:
                ab_node = BINOP(ADD_OP(), Number(a), Number(b))
                abc_node = BINOP(ADD_OP(), ab_node, Number(c))
                return BINOP(SUB_OP(), abc_node, Number(d))
    
    def _create_bit32_call(self, func_name, a, b):
        """Create a bit32["func"](a, b) call node."""
        # String node requires (s, raw) - pass plain func_name for both
        # to_lua_source handles quoting automatically
        func_index = Index(String(func_name, func_name), Name('bit32'), notation=1)
        return Call(func_index, [Number(a), Number(b)])
    
    def _create_bit32_call_single(self, func_name, a):
        """Create a bit32["func"](a) call node (for bnot)."""
        func_index = Index(String(func_name, func_name), Name('bit32'), notation=1)
        return Call(func_index, [Number(a)])
    
    def _create_bit32_call_node(self, func_name, node_a, node_b):
        """Create a bit32["func"](node_a, node_b) call where args are AST nodes."""
        func_index = Index(String(func_name, func_name), Name('bit32'), notation=1)
        return Call(func_index, [node_a, node_b])

    def generate_junk_node(self):
        """
        Creates a 'junk' control flow block (If statement) that serves as obfuscation noise.
        Uses simple number comparisons that are always false, avoiding constant pool dependencies.
        """
        if not EQ_OP:
            return None 
        
        # Generate two different random numbers for a condition that's always false
        num1 = random.randint(100, 999)
        num2 = random.randint(100, 999)
        while num1 == num2:
            num2 = random.randint(100, 999)

        # Create Condition: num1 == num2 (always false since they're different)
        left = Number(num1)
        right = Number(num2)
             
        if SPECIFIC_BINOP_MODE:
            condition = EQ_OP(left, right)
        else:
            condition = BINOP(EQ_OP(), left, right)

        # Create Body: while true do break end
        if TRUE_NODE:
            true_node = TRUE_NODE()
        else:
            true_node = Name('true')
        
        loop_body = Block([Break()])
        while_loop = While(true_node, loop_body)
        
        # Return If(condition, Block([while_loop]), [])
        return If(condition, Block([while_loop]), [])

    def convert_method_calls(self, node):
        """
        DISABLED: Previously converted method calls (obj:Method(args)) to function calls.
        
        This is now disabled because Roblox Signals, Enums, and many other Roblox objects
        do not support string indexing for their methods. Calling signal["Connect"] returns nil.
        
        Colon calls (obj:Method()) are preserved as-is to maintain Roblox compatibility.
        """
        # Method is intentionally empty - we keep colon calls as-is
        pass

    def collect_member_names(self, node):
        """
        Pre-pass to collect all member access names (obj.prop) into the constant pool
        WITHOUT transforming the AST. This ensures indices are stable before transform_ast runs.
        """
        if node is None:
            return

        for attr_name, attr_value in list(node.__dict__.items()):
            if attr_name.startswith('_'):
                continue

            if isinstance(attr_value, list):
                for item in attr_value:
                    if isinstance(item, Node):
                        # Check for member access (Index node with dot notation)
                        if isinstance(item, Index) and getattr(item, 'notation', None) == 0:
                            if isinstance(item.idx, Name):
                                key_string = item.idx.id
                                if key_string not in self.constants:
                                    self.constants.append(key_string)
                        # Recurse
                        self.collect_member_names(item)

            elif isinstance(attr_value, Node):
                if isinstance(attr_value, Index) and getattr(attr_value, 'notation', None) == 0:
                    if isinstance(attr_value.idx, Name):
                        key_string = attr_value.idx.id
                        if key_string not in self.constants:
                            self.constants.append(key_string)
                # Recurse
                self.collect_member_names(attr_value)

    def obfuscate_member_access(self, node):
        """
        Traverses the AST to find member access (obj.prop) and converts it
        to table lookup (obj["prop"]) using the constant pool.
        Note: collect_member_names must be called first to populate the pool.
        """
        if node is None:
            return

        # Iterate over all attributes of the node
        for attr_name, attr_value in list(node.__dict__.items()):
            # Skip private attributes
            if attr_name.startswith('_'):
                continue

            # Handle Lists (Code Blocks)
            if isinstance(attr_value, list):
                for item in attr_value:
                    if isinstance(item, Node):
                        # Check if the item itself is a Member Access (Index node with notation 0)
                        if isinstance(item, Index) and getattr(item, 'notation', None) == 0:
                            # Extract the Key (e.g., "prop" from obj.prop)
                            # In dot notation, idx is a Name node.
                            if isinstance(item.idx, Name):
                                key_string = item.idx.id
                                
                                # Get index from Constant Pool (already added by collect_member_names)
                                if key_string in self.constants:
                                    pool_idx = self.constants.index(key_string)
                                    
                                    # Create Lookup Node (_MP[expression])
                                    lookup_node = self.create_lookup_node(pool_idx)
                                    
                                    # Modify the Index Node in-place
                                    item.idx = lookup_node
                                    item.notation = 1 # Switch to Bracket notation
                        
                        # Recurse into the item (to handle nested accesses like obj.a.b)
                        self.obfuscate_member_access(item)

            # Handle Single Nodes
            elif isinstance(attr_value, Node):
                # Check if the attribute itself is a Member Access
                if isinstance(attr_value, Index) and getattr(attr_value, 'notation', None) == 0:
                     # Extract Key
                     if isinstance(attr_value.idx, Name):
                        key_string = attr_value.idx.id
                        
                        # Get index from Pool (already added by collect_member_names)
                        if key_string in self.constants:
                            pool_idx = self.constants.index(key_string)
                            
                            # Create Lookup
                            lookup_node = self.create_lookup_node(pool_idx)
                            
                            # Modify In-Place
                            attr_value.idx = lookup_node
                            attr_value.notation = 1
                
                # Recurse
                self.obfuscate_member_access(attr_value)

    def flatten_root_flow(self, node):
        """
        Flattens the control flow of the root chunk using a state machine.
        This restructures sequential code into a while-loop switch statement.
        """
        # Ensure node has a body (is a Chunk or Block)
        if not hasattr(node, 'body') or not node.body:
            return
            
        # SAFETY CHECK: Must have EQ_OP. BinOp is optional if in specific mode.
        if not EQ_OP:
            print("[-] Warning: Skipping Control Flow Flattening (Eq node not found)")
            return
        
        # If in Generic Mode, we need BINOP container
        if not SPECIFIC_BINOP_MODE and not BINOP:
             print("[-] Warning: Skipping Control Flow Flattening (BinOp container not found in Generic Mode)")
             return

        # Handle Block wrapping (Chunk -> Block -> List)
        # luaparser Chunk.body is usually a Block object. Block.body is the list.
        statements_list = node.body
        if isinstance(node.body, Block):
            statements_list = node.body.body
        
        # Safety check: ensure we have a list to iterate
        if not isinstance(statements_list, list):
            return

        # ---------------------------------------------------------------------
        # PRE-PASS: Hoist Local Definitions
        # We must move all root-level local definitions (LocalAssign, LocalFunction)
        # to the very top, outside the state machine loop. Otherwise, variables defined
        # in State 1 will be out of scope (nil) in State 2.
        # ---------------------------------------------------------------------
        hoisted_names = []
        new_statements_list = []
        
        for stmt in statements_list:
            if isinstance(stmt, LocalAssign):
                # Collect names to hoist
                for target in stmt.targets:
                    if isinstance(target, Name):
                        # Create a new Name node for the declaration to avoid reference issues
                        hoisted_names.append(Name(target.id))
                
                # Convert definition to assignment: local x = 1 -> x = 1
                # If there are values, we keep the assignment.
                if stmt.values:
                    new_stmt = Assign(stmt.targets, stmt.values)
                    new_statements_list.append(new_stmt)
                # If "local x" (no values), we just remove the statement as hoisting handles the declaration
                
            elif isinstance(stmt, LocalFunction):
                # Hoist function name
                if isinstance(stmt.name, Name):
                    hoisted_names.append(Name(stmt.name.id))
                    
                    # Convert: local function f() end -> f = function() end
                    # Use discovered ANON_FUNC class
                    if ANON_FUNC:
                        anon_func = ANON_FUNC(stmt.args, stmt.body)
                        new_stmt = Assign([stmt.name], [anon_func])
                        new_statements_list.append(new_stmt)
                    else:
                        # Fallback if discovery failed: keep original stmt (won't be flattened correctly but avoids crash)
                        new_statements_list.append(stmt)
                else:
                    new_statements_list.append(stmt)
            else:
                new_statements_list.append(stmt)

        # Update statements_list to the version where locals are converted to assignments
        statements_list = new_statements_list

        # Step A: Chunking
        # Group statements into small blocks or isolated functions
        chunks = []
        current_chunk = []
        
        for stmt in statements_list:
            # We treat functions as atomic chunks
            if isinstance(stmt, Function): # Note: LocalFunctions are now Assigns containing Functions
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []
                chunks.append([stmt])
            else:
                current_chunk.append(stmt)
                # Group 2-3 statements per chunk for the control flow
                if len(current_chunk) >= 3:
                    chunks.append(current_chunk)
                    current_chunk = []
        
        # Append any remaining statements
        if current_chunk:
            chunks.append(current_chunk)

        # If there are no chunks or just one, flattening adds overhead without much benefit
        # But we proceed if > 0 to test functionality
        if not chunks:
            return

        # Step B: State Assignment
        # Assign a random unique ID to each chunk
        available_states = list(range(1000, 9999))
        random.shuffle(available_states)
        
        chunk_states = []
        for _ in chunks:
            if not available_states:
                # Fallback if we run out of unique states (unlikely for root)
                available_states = list(range(10000, 20000))
            chunk_states.append(available_states.pop())
            
        start_state = chunk_states[0]
        end_state = -1

        # Step C: Build the Dispatcher (Control Flow Graph)
        state_var_name = "_state"
        state_var = Name(state_var_name)
        
        # 1. Initial Assignment: local _state = start_state
        init_assign = LocalAssign([Name(state_var_name)], [Number(start_state)])

        # 2. Construct the If-ElseIf Chain
        # We build this backwards to nest the 'Else' blocks correctly.
        # The base case is the termination check: if _state == -1 then break end
        
        # Build Base Condition
        if SPECIFIC_BINOP_MODE:
            final_cond = EQ_OP(state_var, Number(end_state))
        else:
            final_cond = BINOP(EQ_OP(), state_var, Number(end_state))

        dispatch_chain = If(
            final_cond,
            Block([Break()]),
            [] # Empty else block for the final check
        )
        
        # Iterate backwards through chunks
        for i in range(len(chunks) - 1, -1, -1):
            curr_state = chunk_states[i]
            
            # Determine next state
            if i < len(chunks) - 1:
                next_state = chunk_states[i+1]
            else:
                next_state = end_state
                
            # Prepare Block: Statements + State Transition
            stmts = list(chunks[i])  # Make a copy to avoid modifying original
            # Add state transition: _state = next_state
            # Note: This is a normal Assign, not LocalAssign, as it updates the existing var
            state_transition = Assign([Name(state_var_name)], [Number(next_state)])
            
            # CRITICAL FIX: Check if the last statement is a Return
            # In Lua, 'return' must be the last statement in a block
            # So we insert the state transition BEFORE the return, not after
            if stmts and isinstance(stmts[-1], Return):
                # Pop return, add state transition, put return back
                return_stmt = stmts.pop()
                stmts.append(state_transition)
                stmts.append(return_stmt)
            else:
                # Normal case: append state update at the end
                stmts.append(state_transition)
            
            # Combine statements into block
            block_body = Block(stmts)
            
            # Condition: _state == curr_state
            if SPECIFIC_BINOP_MODE:
                condition = EQ_OP(state_var, Number(curr_state))
            else:
                condition = BINOP(EQ_OP(), state_var, Number(curr_state))
            
            # Wrap in If node
            # The previous chain becomes the 'Else' block of this new If node
            # This creates the effect of: if ... then ... else if ... then ...
            dispatch_chain = If(condition, block_body, [dispatch_chain])
            
        # 3. Create the Loop: while true do [dispatch_chain] end
        
        # Check if TRUE_NODE is a class (callable) or just None
        if TRUE_NODE:
            true_node = TRUE_NODE()
        else:
            # Fallback to a Name node 'true' which Lua treats as the boolean true
            true_node = Name('true')
            
        loop_node = While(true_node, Block([dispatch_chain]))
        
        # Step D: Replacement
        
        # Final Body construction:
        # 1. Hoisted Declarations (local a, b, c)
        # 2. State Init (local _state = ...)
        # 3. Loop
        
        final_body = [init_assign, loop_node]
        
        if hoisted_names:
            hoist_decl = LocalAssign(hoisted_names, []) # local a, b, c
            final_body.insert(0, hoist_decl)
            
        # Replace the original root body with our State Machine structure
        if isinstance(node.body, Block):
            node.body.body = final_body
        else:
            node.body = final_body

    def inject_junk_code(self, node, is_root=True):
        """
        Recursively traverse the AST and inject junk code into statement lists.
        
        Args:
            node: The current AST node.
            is_root (bool): True if this is the top-level node (Chunk).
        """
        if node is None:
            return

        # Iterate over all attributes of the node
        # We use list(items) to safely modify the object while iterating
        for attr_name, attr_value in list(node.__dict__.items()):
            # Skip private attributes
            if attr_name.startswith('_'):
                continue

            # If the attribute is a list (likely a block of statements OR a list of expressions)
            if isinstance(attr_value, list):
                # Recurse first into children
                for item in attr_value:
                    if isinstance(item, Node):
                        self.inject_junk_code(item, is_root=False)

                # IMPORTANT FIX: Only inject junk code (which are Statements) into 
                # lists that are actually Statement Blocks (usually named 'body').
                # Injecting statements into 'values' (assignments) or 'args' (calls) 
                # creates invalid syntax like "local x = val, if ... end"
                if attr_name == 'body':
                    
                    # Optional: Skip root body injection if desired (handled by control flow flattening anyway)
                    # if is_root: continue 

                    new_list = []
                    for item in attr_value:
                        new_list.append(item)
                        
                        # Roll the dice for junk injection (20% chance)
                        # Ensure we don't inject after returns or breaks
                        if not isinstance(item, (Return, Break)) and random.random() < 0.2:
                            junk_node = self.generate_junk_node()
                            if junk_node:
                                new_list.append(junk_node)
                    
                    # Replace the original list with the new list containing junk
                    setattr(node, attr_name, new_list)

            # If the attribute is a single Node, just recurse
            elif isinstance(attr_value, Node):
                self.inject_junk_code(attr_value, is_root=False)

    def encrypt_string(self, text):
        """
        Encrypts a string into a list of XOR-ed integers.
        
        Args:
            text (str): The raw string to encrypt.
            
        Returns:
            list: A list of integers where each byte is XOR'd with (key + index) % 256.
        """
        encrypted = []
        for i, char in enumerate(text):
            # Mix index 'i' into the key for position-dependent encryption
            # Apply modulo 256 to the key BEFORE XOR to keep it byte-sized
            # XOR of two bytes always produces a valid byte (0-255)
            k = (self.xor_key + i) % 256
            enc_byte = ord(char) ^ k
            encrypted.append(enc_byte)
        return encrypted

    def minify_source(self, source):
        """
        Minifies Lua source code by removing comments, empty lines, and indentation.
        Preserves string literals that may contain special characters.
        
        Args:
            source (str): The Lua source code to minify.
            
        Returns:
            str: The minified source code.
        """
        result = []
        i = 0
        length = len(source)
        
        while i < length:
            # Check for string literals (single or double quotes)
            if source[i] in ('"', "'"):
                quote = source[i]
                result.append(source[i])
                i += 1
                # Copy everything until closing quote, handling escapes
                while i < length:
                    if source[i] == '\\' and i + 1 < length:
                        # Escaped character - copy both
                        result.append(source[i])
                        result.append(source[i + 1])
                        i += 2
                    elif source[i] == quote:
                        result.append(source[i])
                        i += 1
                        break
                    else:
                        result.append(source[i])
                        i += 1
            # Check for long string literals [[ ]] or [=[ ]=]
            elif source[i] == '[' and i + 1 < length and source[i + 1] in ('=', '['):
                # Find the pattern [[, [=[, [==[, etc.
                start = i
                i += 1
                equals = 0
                while i < length and source[i] == '=':
                    equals += 1
                    i += 1
                if i < length and source[i] == '[':
                    i += 1
                    # Now find matching ]=*]
                    closing = ']' + '=' * equals + ']'
                    end_pos = source.find(closing, i)
                    if end_pos != -1:
                        result.append(source[start:end_pos + len(closing)])
                        i = end_pos + len(closing)
                    else:
                        result.append(source[start:i])
                else:
                    result.append(source[start:i])
            # Check for single-line comments
            elif source[i:i+2] == '--':
                # Check for long comment --[[ ]]
                if source[i:i+4] == '--[[' or (source[i:i+3] == '--[' and i + 3 < length and source[i+3] == '='):
                    # Long comment - find closing
                    start = i + 2
                    i += 3
                    equals = 0
                    while i < length and source[i] == '=':
                        equals += 1
                        i += 1
                    if i < length and source[i] == '[':
                        i += 1
                        closing = ']' + '=' * equals + ']'
                        end_pos = source.find(closing, i)
                        if end_pos != -1:
                            i = end_pos + len(closing)
                        # Skip the entire long comment
                    else:
                        # Not a valid long comment, treat as single-line
                        while i < length and source[i] != '\n':
                            i += 1
                else:
                    # Single-line comment - skip to end of line
                    while i < length and source[i] != '\n':
                        i += 1
            else:
                result.append(source[i])
                i += 1
        
        # Now process the comment-free source
        source = ''.join(result)
        
        # Split into lines, strip whitespace, and filter empty lines
        lines = source.split('\n')
        lines = [line.strip() for line in lines]
        lines = [line for line in lines if line]  # Remove empty lines
        
        return '\n'.join(lines)

    def preprocess_luau(self, source):
        """
        Preprocesses LuaU-specific syntax that luaparser doesn't support.
        Converts LuaU extensions to standard Lua equivalents.
        
        Args:
            source (str): The LuaU source code.
            
        Returns:
            str: Source code compatible with standard Lua parser.
        """
        # Handle 'continue' keyword - not supported by luaparser
        # We need to be careful to only replace 'continue' as a statement, not inside strings
        result = []
        i = 0
        length = len(source)
        
        while i < length:
            # Check for string literals - skip them entirely
            if source[i] in ('"', "'"):
                quote = source[i]
                result.append(source[i])
                i += 1
                while i < length:
                    if source[i] == '\\' and i + 1 < length:
                        result.append(source[i])
                        result.append(source[i + 1])
                        i += 2
                    elif source[i] == quote:
                        result.append(source[i])
                        i += 1
                        break
                    else:
                        result.append(source[i])
                        i += 1
            # Check for long strings [[ ]]
            elif source[i] == '[' and i + 1 < length and source[i + 1] in ('=', '['):
                start = i
                i += 1
                equals = 0
                while i < length and source[i] == '=':
                    equals += 1
                    i += 1
                if i < length and source[i] == '[':
                    i += 1
                    closing = ']' + '=' * equals + ']'
                    end_pos = source.find(closing, i)
                    if end_pos != -1:
                        result.append(source[start:end_pos + len(closing)])
                        i = end_pos + len(closing)
                    else:
                        result.append(source[start:i])
                else:
                    result.append(source[start:i])
            # Check for comments
            elif source[i:i+2] == '--':
                start = i
                if source[i:i+4] == '--[[' or (source[i:i+3] == '--[' and i + 3 < length and source[i+3] == '='):
                    # Long comment
                    i += 3
                    equals = 0
                    while i < length and source[i] == '=':
                        equals += 1
                        i += 1
                    if i < length and source[i] == '[':
                        i += 1
                        closing = ']' + '=' * equals + ']'
                        end_pos = source.find(closing, i)
                        if end_pos != -1:
                            result.append(source[start:end_pos + len(closing)])
                            i = end_pos + len(closing)
                        else:
                            result.append(source[start:])
                            i = length
                    else:
                        while i < length and source[i] != '\n':
                            i += 1
                        result.append(source[start:i])
                else:
                    # Single-line comment - preserve it
                    while i < length and source[i] != '\n':
                        i += 1
                    result.append(source[start:i])
            # Check for 'continue' keyword
            elif source[i:i+8] == 'continue':
                # Make sure it's not part of a larger identifier
                before_ok = (i == 0 or not source[i-1].isalnum() and source[i-1] != '_')
                after_pos = i + 8
                after_ok = (after_pos >= length or not source[after_pos].isalnum() and source[after_pos] != '_')
                
                if before_ok and after_ok:
                    # Replace 'continue' with a goto-based equivalent that luaparser can handle
                    # We use a special comment marker that we'll need to handle, or just remove it
                    # Since luaparser doesn't support goto either, we'll use a dummy that does nothing
                    # but preserves the structure: 'do end' (empty block)
                    result.append('do --[[CONTINUE]] end')
                    i += 8
                else:
                    result.append(source[i])
                    i += 1
            else:
                result.append(source[i])
                i += 1
        
        return ''.join(result)
    
    def load_source(self):
        """
        Read the input Lua file and parse it into an Abstract Syntax Tree.
        
        The AST allows us to programmatically traverse and transform the code
        structure without dealing with raw text manipulation.
        
        Raises:
            FileNotFoundError: If input_file does not exist
            luaparser.ParseError: If the Lua syntax is invalid
        """
        with open(self.input_file, 'r', encoding='utf-8') as f:
            source_code = f.read()

        # Preprocess LuaU-specific syntax (like 'continue') before parsing
        source_code = self.preprocess_luau(source_code)

        # Construct the Injection String
        # Start with the Security Check
        injection = "if not string.find(identifyexecutor(), 'Matcha') then while true do end end\n"
        
        # Add the Environment Fetcher (Safe fallback)
        injection += "local _ENV = (getgenv and getgenv()) or getfenv() or _G\n"
        
        # Loop through self.api_globals and create proxy definitions
        for api in self.api_globals:
            injection += f"local {api} = _ENV['{api}']\n"
            
        # Prepend to Source
        source_code = injection + "\n" + source_code
        
        # Parse source into AST using luaparser
        # This gives us a tree structure we can walk and modify
        self.ast = ast.parse(source_code)
    
    def extract_constants(self, node):
        """
        Recursively walk the AST and extract all string/number constants.
        
        This is the "walker" - it digs through every node in the tree,
        finds literals (strings and numbers), and adds them to our constant pool.
        Later, these will be replaced with table lookups to hide the actual values.
        
        Args:
            node: An AST node to process (can be any node type)
        """
        # Base case: if node is None or not an AST node, skip it
        if node is None:
            return
        
        # Check if this node is a String literal
        # luaparser uses 'String' node type with '.s' attribute for the value
        if isinstance(node, String):
            string_value = node.s
            
            # SAFEGUARD: Ensure it's a native string to avoid isinstance issues later
            if not isinstance(string_value, str):
                string_value = str(string_value)
                
            # Only add if not already in constant pool (avoid duplicates)
            if string_value not in self.constants:
                self.constants.append(string_value)
                # Commenting out verbose print for cleaner output, as requested implicitly by just printing final size
                # print(f'[+] Lifted Constant: "{string_value}"')
            return
        
        # Check if this node is a Number literal
        # luaparser uses 'Number' node type with '.n' attribute for the value
        if isinstance(node, Number):
            number_value = node.n
            # Only add if not already in constant pool (avoid duplicates)
            if number_value not in self.constants:
                self.constants.append(number_value)
                # print(f'[+] Lifted Constant: {number_value}')
            return
        
        # Recursion: Process all child nodes
        # AST nodes have various attributes that can contain child nodes.
        # We iterate over __dict__ to find child nodes or lists of nodes.
        
        if hasattr(node, '__dict__'):
            for attr_name, attr_value in node.__dict__.items():
                # Skip private/internal attributes
                if attr_name.startswith('_'):
                    continue
                
                # Recurse into list of nodes
                if isinstance(attr_value, list):
                    for item in attr_value:
                        if isinstance(item, Node):
                            self.extract_constants(item)
                
                # Recurse into single node
                elif isinstance(attr_value, Node):
                    self.extract_constants(attr_value)

    def create_lookup_node(self, index):
        """
        Helper method that generates the AST structure for _MP[x].

        Args:
            index (int): The 0-based index of the constant in the pool.

        Returns:
            Index: An AST node representing the table lookup.
        """
        # Lua is 1-based, so add 1 to the index.
        lua_index = index + 1
        
        # Create a Name node using self.pool_name (e.g., _MP)
        table_node = Name(self.pool_name)

        # Generate a Mutated Expression for the index
        # Instead of just Number(lua_index), we perform math to get there.
        index_node = self.generate_mutated_expression(lua_index)

        # Return an Index node. 
        # luaparser Index signature: (idx, value, notation)
        # We want _MP[expression], so:
        # idx = index_node (the math expression)
        # value = table_node (the table name _MP)
        # notation = 1 (Bracket notation)
        return Index(index_node, table_node, notation=1)

    def rename_variables(self, node, scope=None):
        """
        Recursively walks the AST to rename local variables to obfuscated names.
        Maintains lexical scoping rules to ensure variable references remain valid.

        Args:
            node: The current AST node.
            scope (dict): A dictionary mapping original names to obfuscated names.
        """
        if node is None:
            return
        if scope is None:
            scope = {}

        # 1. Definitions: Local Assign (e.g., local x = 5)
        if isinstance(node, LocalAssign):
            # Process values (RHS) using CURRENT scope (before redefinition)
            for value in node.values:
                self.rename_variables(value, scope)
            # Process targets (LHS) - Define new vars in CURRENT scope
            for target in node.targets:
                if isinstance(target, Name):
                    new_name = self.generate_var_name()
                    scope[target.id] = new_name
                    target.id = new_name
            return

        # 2. Definitions: Local Function (e.g., local function foo())
        if isinstance(node, LocalFunction):
            # Name is defined in CURRENT scope
            if isinstance(node.name, Name):
                new_name = self.generate_var_name()
                scope[node.name.id] = new_name
                node.name.id = new_name
            
            # Arguments and Body are in INNER scope
            inner_scope = scope.copy()
            for arg in node.args:
                if isinstance(arg, Name):
                    new_arg = self.generate_var_name()
                    inner_scope[arg.id] = new_arg
                    arg.id = new_arg
            
            self.rename_variables(node.body, inner_scope)
            return

        # 3. Global/Anonymous Functions (e.g., function(x) ... end, function Tbl.method() ... end)
        if isinstance(node, Function):
            # Handle function name - for functions like "function UILib.new()" 
            # the name is an Index node where the table (UILib) may need renaming
            func_name = getattr(node, 'name', None)
            if func_name:
                # Rename any Name references in the function name using current scope
                self.rename_variables(func_name, scope)
            
            # Args start new inner scope
            inner_scope = scope.copy()
            for arg in node.args:
                if isinstance(arg, Name):
                    new_arg = self.generate_var_name()
                    inner_scope[arg.id] = new_arg
                    arg.id = new_arg
            
            self.rename_variables(node.body, inner_scope)
            return

        # 4. Numeric For Loops
        # Check against resolved NUMERIC_FOR type
        if NUMERIC_FOR and isinstance(node, NUMERIC_FOR):
            # Safely get attributes using getattr, defaulting to None if missing
            start_node = getattr(node, 'start', None)
            end_node = getattr(node, 'stop', getattr(node, 'end', None))
            step_node = getattr(node, 'step', None)
            
            # Rename children if they are valid AST Nodes (ignore ints/primitives)
            if isinstance(start_node, Node): self.rename_variables(start_node, scope)
            if isinstance(end_node, Node): self.rename_variables(end_node, scope)
            if isinstance(step_node, Node): self.rename_variables(step_node, scope)
            
            inner_scope = scope.copy()
            # Loop variable might be 'var', 'target', or 'name' depending on luaparser version
            loop_var = getattr(node, 'target', getattr(node, 'var', getattr(node, 'name', None)))
            if loop_var and isinstance(loop_var, Name):
                new_name = self.generate_var_name()
                inner_scope[loop_var.id] = new_name
                loop_var.id = new_name
            
            # Handle Body: Might be a list or a single Node
            body_node = getattr(node, 'body', [])
            if isinstance(body_node, list):
                for child in body_node:
                    self.rename_variables(child, inner_scope)
            elif isinstance(body_node, Node):
                self.rename_variables(body_node, inner_scope)
            return

        # 5. Generic For Loops
        # Check against resolved GENERIC_FOR type
        if GENERIC_FOR and isinstance(node, GENERIC_FOR):
            # Handle iterators (might be 'iter', 'iters', or 'iterators')
            iter_nodes = getattr(node, 'iters', getattr(node, 'iter', getattr(node, 'iterators', [])))
            if isinstance(iter_nodes, list):
                for item in iter_nodes:
                    self.rename_variables(item, scope)
            elif isinstance(iter_nodes, Node):
                self.rename_variables(iter_nodes, scope)
            
            inner_scope = scope.copy()
            # Handle loop variable names (might be 'names', 'targets', or 'vars')
            names_list = getattr(node, 'targets', getattr(node, 'names', getattr(node, 'vars', [])))
            if isinstance(names_list, list):
                for name in names_list:
                    if isinstance(name, Name):
                        new_name = self.generate_var_name()
                        inner_scope[name.id] = new_name
                        name.id = new_name
            
            # Handle Body
            body_node = getattr(node, 'body', [])
            if isinstance(body_node, list):
                for child in body_node:
                    self.rename_variables(child, inner_scope)
            elif isinstance(body_node, Node):
                self.rename_variables(body_node, inner_scope)
            return

        # 6. Control Flow Blocks (Scope Copying)
        # While Loop
        if isinstance(node, While):
            # Try getting 'condition' or 'test' (version dependent)
            condition_node = getattr(node, 'condition', getattr(node, 'test', None))
            if condition_node:
                self.rename_variables(condition_node, scope) 

            inner_scope = scope.copy()
            
            # Handle Body (might be list or single node)
            body_node = getattr(node, 'body', [])
            if isinstance(body_node, list):
                for child in body_node:
                    self.rename_variables(child, inner_scope)
            elif isinstance(body_node, Node):
                self.rename_variables(body_node, inner_scope)
            return

        # Repeat Loop (Special: Condition uses inner scope)
        if isinstance(node, Repeat):
            inner_scope = scope.copy()
            
            # Handle Body
            body_node = getattr(node, 'body', [])
            if isinstance(body_node, list):
                for child in body_node:
                    self.rename_variables(child, inner_scope)
            elif isinstance(body_node, Node):
                self.rename_variables(body_node, inner_scope)
            
            # Condition is evaluated in inner scope for Repeat loops
            condition_node = getattr(node, 'condition', getattr(node, 'test', None))
            if condition_node:
                self.rename_variables(condition_node, inner_scope)
            return

        # Do Block
        if isinstance(node, Do):
            inner_scope = scope.copy()
            
            # Handle Body
            body_node = getattr(node, 'body', [])
            if isinstance(body_node, list):
                for child in body_node:
                    self.rename_variables(child, inner_scope)
            elif isinstance(body_node, Node):
                self.rename_variables(body_node, inner_scope)
            return
            
        # If Statement
        if isinstance(node, If):
            # Try getting 'condition' or 'test'
            condition_node = getattr(node, 'condition', getattr(node, 'test', None))
            if condition_node:
                self.rename_variables(condition_node, scope)
            
            then_scope = scope.copy()
            # Handle Body
            body_node = getattr(node, 'body', [])
            if isinstance(body_node, list):
                for child in body_node:
                    self.rename_variables(child, then_scope)
            elif isinstance(body_node, Node):
                self.rename_variables(body_node, then_scope)
            
            # Else/ElseIf handling
            else_node = getattr(node, 'elsebody', getattr(node, 'orelse', None))
            if else_node:
                else_scope = scope.copy() # Else gets its own fresh copy of the OUTER scope
                if isinstance(else_node, list):
                     for child in else_node:
                        self.rename_variables(child, else_scope)
                elif isinstance(else_node, Node):
                    self.rename_variables(else_node, else_scope)
            return

        # 7. Usage: Name Nodes
        if isinstance(node, Name):
            # If it's in our scope, rename it. If not, it's global or upvalue.
            if node.id in scope:
                node.id = scope[node.id]
            return

        # 8. Index Nodes (Member Access) - Special handling
        # For dot notation (math.max), we must NOT rename the member name (max)
        # Only rename the table part (math) if it's a local variable
        if isinstance(node, Index):
            # Rename the table/value part (e.g., 'math' in math.max, or 'obj' in obj.prop)
            value_node = getattr(node, 'value', None)
            if value_node and isinstance(value_node, Node):
                self.rename_variables(value_node, scope)
            
            # For bracket notation (obj[key] or obj["key"]), rename the index expression
            # For dot notation (obj.key), do NOT rename the key - it's a member name, not a variable
            # notation can be an enum (IndexNotation.DOT/SQUARE) or an int (0/1)
            notation = getattr(node, 'notation', None)
            
            # Detect if this is DOT notation (where we should NOT rename the idx)
            is_dot_notation = False
            if notation is not None:
                # Handle both enum and int representations
                notation_str = str(notation)
                if 'DOT' in notation_str or notation == 0:
                    is_dot_notation = True
            
            idx_node = getattr(node, 'idx', None)
            if idx_node and isinstance(idx_node, Node):
                if not is_dot_notation:
                    # Bracket notation - rename variable references in the index
                    self.rename_variables(idx_node, scope)
                # If dot notation, skip - it's a property name, not a variable
            return

        # 9. General Recursion for other nodes (Block, Chunk, Return, Call, etc.)
        for attr_name, attr_value in list(node.__dict__.items()):
            if attr_name.startswith('_'): continue
            
            if isinstance(attr_value, list):
                for item in attr_value:
                    if isinstance(item, Node):
                        self.rename_variables(item, scope)
            elif isinstance(attr_value, Node):
                self.rename_variables(attr_value, scope)

    def transform_ast(self, node):
        """Traverse the AST and replace String/Number literals with constant-pool lookups."""
        if node is None:
            return

        # Only AST nodes have __dict__
        if not hasattr(node, "__dict__"):
            return

        for attr_name, attr_value in list(node.__dict__.items()):
            if attr_name.startswith("_"):
                continue

            # Lists (blocks, args, etc.)
            if isinstance(attr_value, list):
                for i, item in enumerate(attr_value):
                    if isinstance(item, (String, Number)):
                        val = item.s if isinstance(item, String) else item.n
                        if val in self.constants:
                            idx = self.constants.index(val)
                            attr_value[i] = self.create_lookup_node(idx)
                    elif isinstance(item, Node):
                        self.transform_ast(item)

            # Single child node
            elif isinstance(attr_value, Node):
                if isinstance(attr_value, (String, Number)):
                    val = attr_value.s if isinstance(attr_value, String) else attr_value.n
                    if val in self.constants:
                        idx = self.constants.index(val)
                        setattr(node, attr_name, self.create_lookup_node(idx))
                else:
                    self.transform_ast(attr_value)

    def generate_output(self):
        """
        Generate the final obfuscated script and write it to the output file.
        
        This method:
        1. Injects a Lua decryptor function at the top.
        2. Constructs the Lua table for the constant pool with encrypted strings.
        3. Converts the modified AST back into Lua source code.
        4. Combines everything and writes to the file.
        """
        
        # Generate randomized decryptor function name
        decryptor_name = self.generate_var_name()
        
        # Define standalone Bit32 implementation
        # Creates its own bit32 table to avoid read-only global issues in Roblox/Matcha
        bit32_polyfill = """
bit32 = {}
local N = 32
local P = 2 ^ N
bit32.bnot = function(x)
    x = x % P
    return (P - 1) - x
end
bit32.band = function(x, y)
    if (y == 255) then return x % 256 end
    if (y == 65535) then return x % 65536 end
    if (y == 4294967295) then return x % 4294967296 end
    x, y = x % P, y % P
    local r = 0
    local p = 1
    for i = 1, N do
        local a, b = x % 2, y % 2
        x, y = math.floor(x / 2), math.floor(y / 2)
        if ((a + b) == 2) then r = r + p end
        p = 2 * p
    end
    return r
end
bit32.bor = function(x, y)
    if (y == 255) then return (x - (x % 256)) + 255 end
    if (y == 65535) then return (x - (x % 65536)) + 65535 end
    if (y == 4294967295) then return 4294967295 end
    x, y = x % P, y % P
    local r = 0
    local p = 1
    for i = 1, N do
        local a, b = x % 2, y % 2
        x, y = math.floor(x / 2), math.floor(y / 2)
        if ((a + b) >= 1) then r = r + p end
        p = 2 * p
    end
    return r
end
bit32.bxor = function(x, y)
    x, y = x % P, y % P
    local r = 0
    local p = 1
    for i = 1, N do
        local a, b = x % 2, y % 2
        x, y = math.floor(x / 2), math.floor(y / 2)
        if ((a + b) == 1) then r = r + p end
        p = 2 * p
    end
    return r
end
bit32.lshift = function(x, s_amount)
    if (math.abs(s_amount) >= N) then return 0 end
    x = x % P
    if (s_amount < 0) then return math.floor(x * (2 ^ s_amount))
    else return (x * (2 ^ s_amount)) % P end
end
bit32.rshift = function(x, s_amount)
    if (math.abs(s_amount) >= N) then return 0 end
    x = x % P
    if (s_amount > 0) then return math.floor(x * (2 ^ -s_amount))
    else return (x * (2 ^ -s_amount)) % P end
end
bit32.arshift = function(x, s_amount)
    if (math.abs(s_amount) >= N) then return 0 end
    x = x % P
    if (s_amount > 0) then
        local add = 0
        if (x >= (P / 2)) then add = P - (2 ^ (N - s_amount)) end
        return math.floor(x * (2 ^ -s_amount)) + add
    else return (x * (2 ^ -s_amount)) % P end
end
"""

        # Define the Lua decryptor function string
        # Uses bit32.bxor for compatibility since '~' operator failed in Matcha VM
        # Index-mixed decryption: key varies by position to match encrypt_string
        decryptor_lua = (
            f"local function {decryptor_name}(bytes)\n"
            f"    local res = {{}}\n"
            f"    for i, b in ipairs(bytes) do\n"
            f"        local k = ({self.xor_key} + (i - 1)) % 256\n"
            f"        table.insert(res, string.char(bit32.bxor(b, k)))\n"
            f"    end\n"
            f"    return table.concat(res)\n"
            f"end\n\n"
        )

        # 1. Generate Constant Table String
        # Start a string variable
        pool_lua = "local " + self.pool_name + " = {"
        
        # Iterate through self.constants
        for const in self.constants:
            # STRICT ORDERING: We must process every constant to maintain index alignment.
            # Do not skip items.
            if isinstance(const, (int, float)) and not isinstance(const, bool):
                pool_lua += str(const) + ","
            elif isinstance(const, str):
                # Encrypt the string
                encrypted_bytes = self.encrypt_string(const)
                # Format list as Lua table: {10, 55, ...}
                bytes_str = str(encrypted_bytes).replace('[', '{').replace(']', '}')
                # Wrap in decryptor call with randomized function name
                pool_lua += f"{decryptor_name}({bytes_str}),"
            else:
                # Fallback: use repr to ensure it is written to the table
                # This catches any unexpected types to prevent index shifting
                pool_lua += repr(const) + ","
                
        # Close the table
        pool_lua += "}"
        
        # 2. Generate the Script Body
        # Use ast.to_lua_source(self.ast) to convert modified tree back to Lua source
        script_lua = ast.to_lua_source(self.ast)
        
        # 3. Combine and Write
        # Order: Polyfill -> Decryptor -> Constant Pool -> Script
        final_output = bit32_polyfill + "\n" + decryptor_lua + pool_lua + "\n\n" + script_lua
        
        # 4. Minify the output (remove comments, empty lines, indentation)
        final_output = self.minify_source(final_output)
        
        # 5. Add fake header to mislead reverse engineers
        fake_header = "-- Roblox Internal Script // Autogenerated\n"
        final_output = fake_header + final_output
        
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                f.write(final_output)
            print(f"[+] Obfuscated code written to: {self.output_file}")
        except Exception as e:
            print(f"[-] Error writing output file: {e}")


# Main Execution Block
if __name__ == "__main__":
    # 1. Instantiate MatchaObfuscator
    obfuscator = MatchaObfuscator("input.lua", "output.lua")
    
    # 2. Call load_source()
    print("[*] Loading source code...")
    try:
        obfuscator.load_source()
        print("[+] Source loaded successfully")
    except FileNotFoundError:
        print("[-] Error: 'input.lua' not found. Please create this file to test.")
        exit(1)
    except Exception as e:
        print(f"[-] Error parsing Lua: {e}")
        exit(1)

    # 3. Convert Method Calls (before extracting constants)
    # This converts obj:Method() -> obj.Method(obj) so method names become member accesses
    print("[*] Converting Method Calls...")
    obfuscator.convert_method_calls(obfuscator.ast)
    print("[*] Method Calls Transformed: obj:Method() -> obj.Method(obj)")

    # 4. Extract member access names into constant pool FIRST
    # We need to collect these before extract_constants so indices are stable
    print("[*] Collecting Member Access Names...")
    obfuscator.collect_member_names(obfuscator.ast)
    
    # 5. Extract constants (strings and numbers from literals)
    print("[*] Extracting constants...")
    obfuscator.extract_constants(obfuscator.ast)
    print(f"[+] Final Constant Pool Size: {len(obfuscator.constants)}")

    # 6. Rename Variables
    print("[*] Renaming local variables....")
    obfuscator.rename_variables(obfuscator.ast)
    print(f"[+] Variables Renamed (Count: {obfuscator.var_count})")

    # 7. Transform AST (Replace literals with _MP[i])
    print("[*] Transforming AST...")
    obfuscator.transform_ast(obfuscator.ast)
    print("[*] AST Transformed: Literals replaced with table lookups")

    # 8. Obfuscate Member Access (now transform the collected names to lookups)
    print("[*] Obfuscating Member Access...")
    obfuscator.obfuscate_member_access(obfuscator.ast)
    print("[*] Member Access Transformed: obj.prop -> obj[_MP[i]]")

    # 9. Inject Junk Code (Post Transformation)
    print("[*] Injecting Junk Code...")
    obfuscator.inject_junk_code(obfuscator.ast)
    print("[+] Junk Code Injected (20% Density)")

    # 10. Flatten Root Flow
    print("[*] Flattening Root Control Flow...")
    obfuscator.flatten_root_flow(obfuscator.ast)
    print("[+] Root Control Flow Flattened")
    
    # 11. Generate Output
    print("[*] Generating output file...")
    obfuscator.generate_output()
    print(f"[+] Obfuscated script written to {obfuscator.output_file}")
