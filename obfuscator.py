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
        
        # Encryption Key for Strings - Dynamic Runtime Key
        # The actual key is computed at runtime using Matcha-specific globals
        # This makes static analysis impossible - dumpers will get garbage
        self.xor_key_base = random.randint(1, 200)  # Base component (visible in code)
        self.xor_key_runtime_mult = random.randint(2, 7)  # Multiplier for runtime component
        self.xor_key_runtime_add = random.randint(10, 50)  # Addition for runtime component
        # The full key formula: (base + (runtime_value * mult + add)) % 256
        # Where runtime_value comes from Matcha-specific globals at runtime
        
        # Constant Pool: All strings and numbers will be extracted here
        # This makes static analysis harder as values are no longer inline
        self.constants = []
        self.pool_name = self.generate_var_name()  # Randomized constant table name

        # List of sensitive globals to protect
        self.api_globals = ["Drawing", "Vector3", "Color3", "Instance", "CFrame", "game", "workspace", "wait", "spawn", "iskeypressed", "mouse1click", "WorldToScreen"]
        
        # AST will be populated by load_source()
        self.ast = None
        
        # Closure-Based Virtualization
        # Maps operation types to their virtualized function keys
        self.virt_table_name = self.generate_var_name()  # Name of the operator table
        self.virt_ops = {}  # Maps (op_type, key_modifier) -> random_key
        self.virt_key_seed = random.randint(1000, 9999)  # Seed for key generation
        
        # =====================================================================
        # HYBRID REGISTER-BASED VM - Instruction Set Architecture (ISA)
        # =====================================================================
        # Opcodes are RANDOMIZED on every obfuscation run for anti-pattern matching
        # Register-based design for better performance than stack-based VMs
        
        # Define all opcode names
        self.vm_opcode_names = [
            'MOV',        # R[A] = R[B] - Register move
            'LOADK',      # R[A] = Constants[K] - Load constant (critical for strings)
            'LOADNIL',    # R[A] = nil
            'LOADBOOL',   # R[A] = (bool)B; if C then IP++
            'GETGLOBAL',  # R[A] = _G[Constants[K]] - Get global (game, workspace, etc)
            'SETGLOBAL',  # _G[Constants[K]] = R[A] - Set global
            'GETTABLE',   # R[A] = R[B][R[C]] - Table index
            'SETTABLE',   # R[A][R[B]] = R[C] - Table set
            'NEWTABLE',   # R[A] = {} - Create table
            'CALL',       # R[A](R[A+1], ..., R[A+B]) - Function call (bridge to Roblox/Matcha)
            'RETURN',     # return R[A], ..., R[A+B] - Return values
            'JMP',        # IP = IP + Offset - Unconditional jump
            'JMPIF',      # if R[A] then IP = IP + Offset - Conditional jump if true
            'JMPIFNOT',   # if not R[A] then IP = IP + Offset - Conditional jump if false
            'EQ',         # if (R[A] == R[B]) ~= C then IP++ - Equality test
            'LT',         # if (R[A] < R[B]) ~= C then IP++ - Less than
            'LE',         # if (R[A] <= R[B]) ~= C then IP++ - Less or equal
            'TEST',       # if not R[A] == C then IP++ - Boolean test
            'FAST_ADD',   # R[A] = R[B] + R[C] - Fast addition
            'FAST_SUB',   # R[A] = R[B] - R[C] - Fast subtraction
            'FAST_MUL',   # R[A] = R[B] * R[C] - Fast multiplication
            'FAST_DIV',   # R[A] = R[B] / R[C] - Fast division
            'FAST_MOD',   # R[A] = R[B] % R[C] - Fast modulo
            'FAST_UNM',   # R[A] = -R[B] - Unary minus
            'FAST_NOT',   # R[A] = not R[B] - Logical not
            'FAST_LEN',   # R[A] = #R[B] - Length operator
            'CONCAT',     # R[A] = R[B] .. R[C] - String concatenation
            'SELF',       # R[A+1] = R[B]; R[A] = R[B][R[C]] - Method call prep
            'CLOSURE',    # R[A] = closure(Proto[B]) - Create closure
            'VARARG',     # R[A], ..., R[A+B] = ... - Vararg handling
            'FORPREP',    # R[A] -= R[A+2]; IP += B - Numeric for prep
            'FORLOOP',    # R[A] += R[A+2]; if R[A] <= R[A+1] then IP += B; R[A+3] = R[A]
            'TFORLOOP',   # R[A+3], ... = R[A](R[A+1], R[A+2]); if R[A+3] ~= nil then R[A+2] = R[A+3]
            'SETLIST',    # R[A][(C-1)*50+i] = R[A+i], 1 <= i <= B
            'NOP',        # No operation (for padding/obfuscation)
            'YIELD',      # Yield VM execution (for wait() calls) - saves state and returns
            'NATIVE_CALL',# Call native (non-virtualized) function by index
            'GETUPVAL',   # R[A] = UpValues[B] - Get upvalue
            'SETUPVAL',   # UpValues[B] = R[A] - Set upvalue
        ]
        
        # Generate RANDOMIZED opcode values (shuffled every run)
        opcode_values = list(range(1, len(self.vm_opcode_names) + 1))
        random.shuffle(opcode_values)
        
        # Create the opcode mapping: name -> randomized numeric value
        self.vm_opcodes = {}
        for i, name in enumerate(self.vm_opcode_names):
            self.vm_opcodes[name] = opcode_values[i]
        
        # Reverse mapping for VM dispatch: numeric value -> handler key
        self.vm_opcode_reverse = {v: k for k, v in self.vm_opcodes.items()}
        
        # VM Register file size (R[0] to R[N-1])
        self.vm_register_count = 64
        
        # VM component names (randomized)
        self.vm_registers_name = self.generate_var_name()    # Register file table
        self.vm_constants_name = self.generate_var_name()    # Constants table
        self.vm_bytecode_name = self.generate_var_name()     # Bytecode array
        self.vm_ip_name = self.generate_var_name()           # Instruction pointer
        self.vm_dispatch_name = self.generate_var_name()     # Dispatch table
        self.vm_execute_name = self.generate_var_name()      # Main execute function
        
        # Track which code sections are virtualized vs native
        self.vm_virtualized_sections = []  # List of bytecode arrays for virtualized code
        self.vm_hot_paths = []             # List of AST nodes to keep native

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

    # =========================================================================
    # HYBRID REGISTER-BASED VM - Core Methods
    # =========================================================================
    
    def print_opcode_mapping(self):
        """Debug helper: Print the randomized opcode mapping."""
        print("[*] VM Opcode Mapping (Randomized):")
        for name, value in sorted(self.vm_opcodes.items(), key=lambda x: x[1]):
            print(f"    {value:3d} -> {name}")
    
    def get_opcode(self, name):
        """Get the randomized opcode value for an instruction name."""
        return self.vm_opcodes.get(name, 0)
    
    def generate_vm_dispatch_table(self):
        """
        Generate the Lua code for the VM dispatch table.
        
        Each opcode maps to a handler function. The opcodes are randomized
        so pattern matching against known VM structures is impossible.
        
        Returns:
            str: Lua code defining the dispatch table
        """
        lines = []
        
        # Create dispatch table with randomized opcode keys
        lines.append(f"local {self.vm_dispatch_name} = {{}}")
        
        # MOV: R[A] = R[B]
        op = self.get_opcode('MOV')
        lines.append(f"{self.vm_dispatch_name}[{op}] = function({self.vm_registers_name}, A, B) {self.vm_registers_name}[A] = {self.vm_registers_name}[B] end")
        
        # LOADK: R[A] = Constants[K]
        op = self.get_opcode('LOADK')
        lines.append(f"{self.vm_dispatch_name}[{op}] = function({self.vm_registers_name}, A, K, {self.vm_constants_name}) {self.vm_registers_name}[A] = {self.vm_constants_name}[K] end")
        
        # LOADNIL: R[A] = nil
        op = self.get_opcode('LOADNIL')
        lines.append(f"{self.vm_dispatch_name}[{op}] = function({self.vm_registers_name}, A) {self.vm_registers_name}[A] = nil end")
        
        # LOADBOOL: R[A] = (bool)B
        op = self.get_opcode('LOADBOOL')
        lines.append(f"{self.vm_dispatch_name}[{op}] = function({self.vm_registers_name}, A, B) {self.vm_registers_name}[A] = B ~= 0 end")
        
        # GETGLOBAL: R[A] = _G[Constants[K]]
        op = self.get_opcode('GETGLOBAL')
        lines.append(f"{self.vm_dispatch_name}[{op}] = function({self.vm_registers_name}, A, K, {self.vm_constants_name}) {self.vm_registers_name}[A] = _G[{self.vm_constants_name}[K]] end")
        
        # SETGLOBAL: _G[Constants[K]] = R[A]
        op = self.get_opcode('SETGLOBAL')
        lines.append(f"{self.vm_dispatch_name}[{op}] = function({self.vm_registers_name}, A, K, {self.vm_constants_name}) _G[{self.vm_constants_name}[K]] = {self.vm_registers_name}[A] end")
        
        # GETTABLE: R[A] = R[B][R[C]]
        op = self.get_opcode('GETTABLE')
        lines.append(f"{self.vm_dispatch_name}[{op}] = function({self.vm_registers_name}, A, B, C) {self.vm_registers_name}[A] = {self.vm_registers_name}[B][{self.vm_registers_name}[C]] end")
        
        # SETTABLE: R[A][R[B]] = R[C]
        op = self.get_opcode('SETTABLE')
        lines.append(f"{self.vm_dispatch_name}[{op}] = function({self.vm_registers_name}, A, B, C) {self.vm_registers_name}[A][{self.vm_registers_name}[B]] = {self.vm_registers_name}[C] end")
        
        # NEWTABLE: R[A] = {}
        op = self.get_opcode('NEWTABLE')
        lines.append(f"{self.vm_dispatch_name}[{op}] = function({self.vm_registers_name}, A) {self.vm_registers_name}[A] = {{}} end")
        
        # FAST_ADD: R[A] = R[B] + R[C]
        op = self.get_opcode('FAST_ADD')
        lines.append(f"{self.vm_dispatch_name}[{op}] = function({self.vm_registers_name}, A, B, C) {self.vm_registers_name}[A] = {self.vm_registers_name}[B] + {self.vm_registers_name}[C] end")
        
        # FAST_SUB: R[A] = R[B] - R[C]
        op = self.get_opcode('FAST_SUB')
        lines.append(f"{self.vm_dispatch_name}[{op}] = function({self.vm_registers_name}, A, B, C) {self.vm_registers_name}[A] = {self.vm_registers_name}[B] - {self.vm_registers_name}[C] end")
        
        # FAST_MUL: R[A] = R[B] * R[C]
        op = self.get_opcode('FAST_MUL')
        lines.append(f"{self.vm_dispatch_name}[{op}] = function({self.vm_registers_name}, A, B, C) {self.vm_registers_name}[A] = {self.vm_registers_name}[B] * {self.vm_registers_name}[C] end")
        
        # FAST_DIV: R[A] = R[B] / R[C]
        op = self.get_opcode('FAST_DIV')
        lines.append(f"{self.vm_dispatch_name}[{op}] = function({self.vm_registers_name}, A, B, C) {self.vm_registers_name}[A] = {self.vm_registers_name}[B] / {self.vm_registers_name}[C] end")
        
        # FAST_MOD: R[A] = R[B] % R[C]
        op = self.get_opcode('FAST_MOD')
        lines.append(f"{self.vm_dispatch_name}[{op}] = function({self.vm_registers_name}, A, B, C) {self.vm_registers_name}[A] = {self.vm_registers_name}[B] % {self.vm_registers_name}[C] end")
        
        # FAST_UNM: R[A] = -R[B]
        op = self.get_opcode('FAST_UNM')
        lines.append(f"{self.vm_dispatch_name}[{op}] = function({self.vm_registers_name}, A, B) {self.vm_registers_name}[A] = -{self.vm_registers_name}[B] end")
        
        # FAST_NOT: R[A] = not R[B]
        op = self.get_opcode('FAST_NOT')
        lines.append(f"{self.vm_dispatch_name}[{op}] = function({self.vm_registers_name}, A, B) {self.vm_registers_name}[A] = not {self.vm_registers_name}[B] end")
        
        # FAST_LEN: R[A] = #R[B]
        op = self.get_opcode('FAST_LEN')
        lines.append(f"{self.vm_dispatch_name}[{op}] = function({self.vm_registers_name}, A, B) {self.vm_registers_name}[A] = #{self.vm_registers_name}[B] end")
        
        # CONCAT: R[A] = R[B] .. R[C]
        op = self.get_opcode('CONCAT')
        lines.append(f"{self.vm_dispatch_name}[{op}] = function({self.vm_registers_name}, A, B, C) {self.vm_registers_name}[A] = {self.vm_registers_name}[B] .. {self.vm_registers_name}[C] end")
        
        # EQ: Compare equality
        op = self.get_opcode('EQ')
        lines.append(f"{self.vm_dispatch_name}[{op}] = function({self.vm_registers_name}, A, B) return {self.vm_registers_name}[A] == {self.vm_registers_name}[B] end")
        
        # LT: Less than
        op = self.get_opcode('LT')
        lines.append(f"{self.vm_dispatch_name}[{op}] = function({self.vm_registers_name}, A, B) return {self.vm_registers_name}[A] < {self.vm_registers_name}[B] end")
        
        # LE: Less or equal
        op = self.get_opcode('LE')
        lines.append(f"{self.vm_dispatch_name}[{op}] = function({self.vm_registers_name}, A, B) return {self.vm_registers_name}[A] <= {self.vm_registers_name}[B] end")
        
        # TEST: Boolean test
        op = self.get_opcode('TEST')
        lines.append(f"{self.vm_dispatch_name}[{op}] = function({self.vm_registers_name}, A) return {self.vm_registers_name}[A] end")
        
        # JMP: Unconditional jump (handled specially in executor, but needs entry)
        op = self.get_opcode('JMP')
        lines.append(f"{self.vm_dispatch_name}[{op}] = function() return true end")
        
        # JMPIF: Jump if true (handler returns condition result)
        op = self.get_opcode('JMPIF')
        lines.append(f"{self.vm_dispatch_name}[{op}] = function({self.vm_registers_name}, A) return {self.vm_registers_name}[A] end")
        
        # JMPIFNOT: Jump if false (handler returns condition result)
        op = self.get_opcode('JMPIFNOT')
        lines.append(f"{self.vm_dispatch_name}[{op}] = function({self.vm_registers_name}, A) return not {self.vm_registers_name}[A] end")
        
        # RETURN: Return from VM (handled specially in executor)
        op = self.get_opcode('RETURN')
        lines.append(f"{self.vm_dispatch_name}[{op}] = function({self.vm_registers_name}, A) return {self.vm_registers_name}[A] end")
        
        # CALL: R[A](R[A+1], ..., R[A+B]) - Function call
        # This is more complex - we build args and call
        op = self.get_opcode('CALL')
        call_args_var = self.generate_var_name()
        lines.append(f"{self.vm_dispatch_name}[{op}] = function({self.vm_registers_name}, A, B, C)")
        lines.append(f"    local {call_args_var} = {{}}")
        lines.append(f"    for i = 1, B do {call_args_var}[i] = {self.vm_registers_name}[A + i] end")
        lines.append(f"    local results = {{{self.vm_registers_name}[A](unpack({call_args_var}))}}")
        lines.append(f"    for i = 1, C do {self.vm_registers_name}[A + i - 1] = results[i] end")
        lines.append(f"end")
        
        # SELF: R[A+1] = R[B]; R[A] = R[B][R[C]] - Method call preparation
        op = self.get_opcode('SELF')
        lines.append(f"{self.vm_dispatch_name}[{op}] = function({self.vm_registers_name}, A, B, C) {self.vm_registers_name}[A+1] = {self.vm_registers_name}[B]; {self.vm_registers_name}[A] = {self.vm_registers_name}[B][{self.vm_registers_name}[C]] end")
        
        # NOP: No operation
        op = self.get_opcode('NOP')
        lines.append(f"{self.vm_dispatch_name}[{op}] = function() end")
        
        return "\n".join(lines) + "\n"
    
    def generate_vm_executor(self):
        """
        Generate the optimized VM execution loop.
        
        Uses inline if/elseif for the most common opcodes to avoid
        function call overhead. Supports YIELD for wait() handling.
        
        Returns:
            str: Lua code for the VM executor function
        """
        # Generate randomized local variable names
        ip_var = self.generate_var_name()
        bc_var = self.generate_var_name()
        k_var = self.generate_var_name()
        r_var = self.generate_var_name()
        op_var = self.generate_var_name()
        a_var = self.generate_var_name()
        b_var = self.generate_var_name()
        c_var = self.generate_var_name()
        state_var = self.generate_var_name()
        native_funcs_var = self.generate_var_name()
        
        # Get randomized opcodes
        OP_MOV = self.get_opcode('MOV')
        OP_LOADK = self.get_opcode('LOADK')
        OP_LOADNIL = self.get_opcode('LOADNIL')
        OP_LOADBOOL = self.get_opcode('LOADBOOL')
        OP_GETGLOBAL = self.get_opcode('GETGLOBAL')
        OP_SETGLOBAL = self.get_opcode('SETGLOBAL')
        OP_GETTABLE = self.get_opcode('GETTABLE')
        OP_SETTABLE = self.get_opcode('SETTABLE')
        OP_NEWTABLE = self.get_opcode('NEWTABLE')
        OP_CALL = self.get_opcode('CALL')
        OP_RETURN = self.get_opcode('RETURN')
        OP_JMP = self.get_opcode('JMP')
        OP_JMPIF = self.get_opcode('JMPIF')
        OP_JMPIFNOT = self.get_opcode('JMPIFNOT')
        OP_EQ = self.get_opcode('EQ')
        OP_LT = self.get_opcode('LT')
        OP_LE = self.get_opcode('LE')
        OP_FAST_ADD = self.get_opcode('FAST_ADD')
        OP_FAST_SUB = self.get_opcode('FAST_SUB')
        OP_FAST_MUL = self.get_opcode('FAST_MUL')
        OP_FAST_DIV = self.get_opcode('FAST_DIV')
        OP_FAST_MOD = self.get_opcode('FAST_MOD')
        OP_FAST_UNM = self.get_opcode('FAST_UNM')
        OP_FAST_NOT = self.get_opcode('FAST_NOT')
        OP_FAST_LEN = self.get_opcode('FAST_LEN')
        OP_CONCAT = self.get_opcode('CONCAT')
        OP_SELF = self.get_opcode('SELF')
        OP_YIELD = self.get_opcode('YIELD')
        OP_NATIVE_CALL = self.get_opcode('NATIVE_CALL')
        OP_NOP = self.get_opcode('NOP')
        OP_TEST = self.get_opcode('TEST')
        
        executor = f'''
local {self.vm_execute_name}
do
    local {state_var} = {{}}
    
    {self.vm_execute_name} = function({bc_var}, {k_var}, {native_funcs_var})
        local {r_var} = {state_var}.R or {{}}
        local {ip_var} = {state_var}.IP or 1
        local {op_var}, {a_var}, {b_var}, {c_var}
        local _len = #{bc_var}
        
        while {ip_var} <= _len do
            {op_var} = {bc_var}[{ip_var}]
            {a_var} = {bc_var}[{ip_var}+1]
            {b_var} = {bc_var}[{ip_var}+2]
            {c_var} = {bc_var}[{ip_var}+3]
            
            -- FAST PATH: Most common opcodes first (inline for speed)
            if {op_var} == {OP_LOADK} then
                {r_var}[{a_var}] = {k_var}[{b_var}]
                {ip_var} = {ip_var} + 4
            elseif {op_var} == {OP_MOV} then
                {r_var}[{a_var}] = {r_var}[{b_var}]
                {ip_var} = {ip_var} + 4
            elseif {op_var} == {OP_FAST_ADD} then
                {r_var}[{a_var}] = {r_var}[{b_var}] + {r_var}[{c_var}]
                {ip_var} = {ip_var} + 4
            elseif {op_var} == {OP_FAST_SUB} then
                {r_var}[{a_var}] = {r_var}[{b_var}] - {r_var}[{c_var}]
                {ip_var} = {ip_var} + 4
            elseif {op_var} == {OP_FAST_MUL} then
                {r_var}[{a_var}] = {r_var}[{b_var}] * {r_var}[{c_var}]
                {ip_var} = {ip_var} + 4
            elseif {op_var} == {OP_FAST_DIV} then
                {r_var}[{a_var}] = {r_var}[{b_var}] / {r_var}[{c_var}]
                {ip_var} = {ip_var} + 4
            elseif {op_var} == {OP_GETTABLE} then
                {r_var}[{a_var}] = {r_var}[{b_var}][{r_var}[{c_var}]]
                {ip_var} = {ip_var} + 4
            elseif {op_var} == {OP_SETTABLE} then
                {r_var}[{a_var}][{r_var}[{b_var}]] = {r_var}[{c_var}]
                {ip_var} = {ip_var} + 4
            elseif {op_var} == {OP_GETGLOBAL} then
                {r_var}[{a_var}] = _G[{k_var}[{b_var}]]
                {ip_var} = {ip_var} + 4
            elseif {op_var} == {OP_SETGLOBAL} then
                _G[{k_var}[{b_var}]] = {r_var}[{a_var}]
                {ip_var} = {ip_var} + 4
            elseif {op_var} == {OP_JMP} then
                {ip_var} = {ip_var} + {b_var} * 4 + 4
            elseif {op_var} == {OP_JMPIF} then
                if {r_var}[{a_var}] then
                    {ip_var} = {ip_var} + {b_var} * 4 + 4
                else
                    {ip_var} = {ip_var} + 4
                end
            elseif {op_var} == {OP_JMPIFNOT} then
                if not {r_var}[{a_var}] then
                    {ip_var} = {ip_var} + {b_var} * 4 + 4
                else
                    {ip_var} = {ip_var} + 4
                end
            elseif {op_var} == {OP_EQ} then
                if ({r_var}[{a_var}] == {r_var}[{b_var}]) then
                    {ip_var} = {ip_var} + 4
                else
                    {ip_var} = {ip_var} + 8
                end
            elseif {op_var} == {OP_LT} then
                if ({r_var}[{a_var}] < {r_var}[{b_var}]) then
                    {ip_var} = {ip_var} + 4
                else
                    {ip_var} = {ip_var} + 8
                end
            elseif {op_var} == {OP_LE} then
                if ({r_var}[{a_var}] <= {r_var}[{b_var}]) then
                    {ip_var} = {ip_var} + 4
                else
                    {ip_var} = {ip_var} + 8
                end
            elseif {op_var} == {OP_CALL} then
                local _f = {r_var}[{a_var}]
                local _args = {{}}
                for _i = 1, {b_var} do _args[_i] = {r_var}[{a_var} + _i] end
                local _results = {{_f(unpack(_args))}}
                for _i = 1, {c_var} do {r_var}[{a_var} + _i - 1] = _results[_i] end
                {ip_var} = {ip_var} + 4
            elseif {op_var} == {OP_RETURN} then
                {state_var}.R = nil
                {state_var}.IP = nil
                if {b_var} > 0 then
                    return {r_var}[{a_var}]
                else
                    return nil
                end
            elseif {op_var} == {OP_YIELD} then
                -- Save state and return control
                {state_var}.R = {r_var}
                {state_var}.IP = {ip_var} + 4
                return "YIELD", {r_var}[{a_var}]
            elseif {op_var} == {OP_NATIVE_CALL} then
                -- Call native function by index (hot path optimization)
                if {native_funcs_var} and {native_funcs_var}[{a_var}] then
                    {r_var}[{a_var}] = {native_funcs_var}[{a_var}]()
                end
                {ip_var} = {ip_var} + 4
            elseif {op_var} == {OP_LOADNIL} then
                {r_var}[{a_var}] = nil
                {ip_var} = {ip_var} + 4
            elseif {op_var} == {OP_LOADBOOL} then
                {r_var}[{a_var}] = {b_var} ~= 0
                if {c_var} ~= 0 then {ip_var} = {ip_var} + 8 else {ip_var} = {ip_var} + 4 end
            elseif {op_var} == {OP_NEWTABLE} then
                {r_var}[{a_var}] = {{}}
                {ip_var} = {ip_var} + 4
            elseif {op_var} == {OP_FAST_MOD} then
                {r_var}[{a_var}] = {r_var}[{b_var}] % {r_var}[{c_var}]
                {ip_var} = {ip_var} + 4
            elseif {op_var} == {OP_FAST_UNM} then
                {r_var}[{a_var}] = -{r_var}[{b_var}]
                {ip_var} = {ip_var} + 4
            elseif {op_var} == {OP_FAST_NOT} then
                {r_var}[{a_var}] = not {r_var}[{b_var}]
                {ip_var} = {ip_var} + 4
            elseif {op_var} == {OP_FAST_LEN} then
                {r_var}[{a_var}] = #{r_var}[{b_var}]
                {ip_var} = {ip_var} + 4
            elseif {op_var} == {OP_CONCAT} then
                {r_var}[{a_var}] = {r_var}[{b_var}] .. {r_var}[{c_var}]
                {ip_var} = {ip_var} + 4
            elseif {op_var} == {OP_SELF} then
                {r_var}[{a_var}+1] = {r_var}[{b_var}]
                {r_var}[{a_var}] = {r_var}[{b_var}][{r_var}[{c_var}]]
                {ip_var} = {ip_var} + 4
            elseif {op_var} == {OP_TEST} then
                if {r_var}[{a_var}] then
                    {ip_var} = {ip_var} + 4
                else
                    {ip_var} = {ip_var} + 8
                end
            elseif {op_var} == {OP_NOP} then
                {ip_var} = {ip_var} + 4
            else
                {ip_var} = {ip_var} + 4
            end
        end
        
        {state_var}.R = nil
        {state_var}.IP = nil
        return {r_var}[0]
    end
end
'''
        return executor
    
    def emit_instruction(self, opcode_name, a=0, b=0, c=0):
        """
        Create a bytecode instruction tuple.
        
        Args:
            opcode_name: Name of the opcode (e.g., 'MOV', 'LOADK')
            a, b, c: Operands (meaning depends on opcode)
            
        Returns:
            tuple: (opcode, a, b, c) ready to be added to bytecode
        """
        return (self.get_opcode(opcode_name), a, b, c)
    
    def bytecode_to_lua_table(self, bytecode):
        """
        Convert a list of bytecode instructions to Lua table syntax.
        
        Args:
            bytecode: List of instruction tuples
            
        Returns:
            str: Lua table literal representing the bytecode
        """
        instructions = []
        for instr in bytecode:
            instructions.append(f"{{{instr[0]},{instr[1]},{instr[2]},{instr[3]}}}")
        return "{" + ",".join(instructions) + "}"

    # =========================================================================
    # VIRTUALIZER - AST to Bytecode Compiler
    # =========================================================================
    
    def init_virtualizer(self):
        """
        Initialize the virtualizer state for compilation.
        Must be called before compile_ast.
        """
        # Bytecode output
        self.vm_bytecode = []
        
        # VM Constants pool (separate from string encryption pool)
        self.vm_constants_pool = []
        
        # Register allocation
        self.vm_next_register = 0
        self.vm_max_register = 0
        
        # Variable to register mapping: {var_name: register_index}
        self.vm_var_registers = {}
        
        # Scope stack for nested blocks
        self.vm_scope_stack = []
        
        # Label system for jumps
        self.vm_labels = {}  # label_name -> instruction_index
        self.vm_pending_jumps = []  # [(instruction_index, label_name), ...]
        self.vm_label_counter = 0
    
    def vm_alloc_register(self):
        """Allocate a new register and return its index."""
        reg = self.vm_next_register
        self.vm_next_register += 1
        if self.vm_next_register > self.vm_max_register:
            self.vm_max_register = self.vm_next_register
        return reg
    
    def vm_free_register(self, count=1):
        """Free the last N allocated registers."""
        self.vm_next_register = max(0, self.vm_next_register - count)
    
    def vm_push_scope(self):
        """Push a new variable scope."""
        self.vm_scope_stack.append({
            'vars': dict(self.vm_var_registers),
            'next_reg': self.vm_next_register
        })
    
    def vm_pop_scope(self):
        """Pop a variable scope, restoring previous state."""
        if self.vm_scope_stack:
            scope = self.vm_scope_stack.pop()
            self.vm_var_registers = scope['vars']
            self.vm_next_register = scope['next_reg']
    
    def vm_add_constant(self, value):
        """
        Add a constant to the VM constants pool.
        Returns the index of the constant.
        """
        if value in self.vm_constants_pool:
            return self.vm_constants_pool.index(value)
        self.vm_constants_pool.append(value)
        return len(self.vm_constants_pool) - 1
    
    def vm_emit(self, opcode_name, a=0, b=0, c=0):
        """
        Emit a bytecode instruction.
        Returns the index of the emitted instruction.
        """
        instr = (self.get_opcode(opcode_name), a, b, c)
        idx = len(self.vm_bytecode)
        self.vm_bytecode.append(instr)
        return idx
    
    def vm_create_label(self):
        """Create a new unique label name."""
        self.vm_label_counter += 1
        return f"_L{self.vm_label_counter}"
    
    def vm_set_label(self, label_name):
        """Set a label at the current bytecode position."""
        self.vm_labels[label_name] = len(self.vm_bytecode)
    
    def vm_emit_jump(self, opcode_name, label_name, a=0):
        """
        Emit a jump instruction to a label (resolved later).
        """
        idx = self.vm_emit(opcode_name, a, 0, 0)  # Offset will be patched
        self.vm_pending_jumps.append((idx, label_name))
        return idx
    
    def vm_resolve_jumps(self):
        """
        Resolve all pending jump offsets after compilation.
        """
        for instr_idx, label_name in self.vm_pending_jumps:
            if label_name in self.vm_labels:
                target_idx = self.vm_labels[label_name]
                # Calculate relative offset
                offset = target_idx - instr_idx - 1
                # Patch the instruction
                old_instr = self.vm_bytecode[instr_idx]
                self.vm_bytecode[instr_idx] = (old_instr[0], old_instr[1], offset, old_instr[3])
    
    def compile_expression(self, node, target_reg=None):
        """
        Compile an expression AST node to bytecode.
        
        Args:
            node: The AST expression node
            target_reg: Optional target register (allocates one if None)
            
        Returns:
            int: Register containing the result
        """
        if target_reg is None:
            target_reg = self.vm_alloc_register()
        
        node_type = type(node).__name__
        
        # Number literal
        if node_type == 'Number':
            const_idx = self.vm_add_constant(node.n)
            self.vm_emit('LOADK', target_reg, const_idx, 0)
            return target_reg
        
        # String literal
        elif node_type == 'String':
            const_idx = self.vm_add_constant(node.s)
            self.vm_emit('LOADK', target_reg, const_idx, 0)
            return target_reg
        
        # Nil literal
        elif node_type == 'Nil':
            self.vm_emit('LOADNIL', target_reg, 0, 0)
            return target_reg
        
        # Boolean literal
        elif node_type in ('TrueExpr', 'FalseExpr'):
            bool_val = 1 if node_type == 'TrueExpr' else 0
            self.vm_emit('LOADBOOL', target_reg, bool_val, 0)
            return target_reg
        
        # Variable reference (Name)
        elif node_type == 'Name':
            var_name = node.id
            if var_name in self.vm_var_registers:
                # Local variable - move from its register
                src_reg = self.vm_var_registers[var_name]
                if src_reg != target_reg:
                    self.vm_emit('MOV', target_reg, src_reg, 0)
                return target_reg
            else:
                # Global variable
                const_idx = self.vm_add_constant(var_name)
                self.vm_emit('GETGLOBAL', target_reg, const_idx, 0)
                return target_reg
        
        # Binary operations
        elif node_type in ('AddOp', 'SubOp', 'MultOp', 'DivOp', 'ModOp'):
            left_reg = self.compile_expression(node.left)
            right_reg = self.compile_expression(node.right)
            
            op_map = {
                'AddOp': 'FAST_ADD',
                'SubOp': 'FAST_SUB',
                'MultOp': 'FAST_MUL',
                'DivOp': 'FAST_DIV',
                'ModOp': 'FAST_MOD'
            }
            self.vm_emit(op_map[node_type], target_reg, left_reg, right_reg)
            
            # Free temporary registers
            if right_reg >= target_reg + 1:
                self.vm_free_register(2)
            return target_reg
        
        # Comparison operations
        elif node_type in ('EqToOp', 'NotEqToOp', 'LessThanOp', 'GreaterThanOp', 'LessOrEqThanOp', 'GreaterOrEqThanOp'):
            left_reg = self.compile_expression(node.left)
            right_reg = self.compile_expression(node.right)
            
            # For comparisons, we use a pattern: compare, then set boolean based on result
            if node_type == 'EqToOp':
                self.vm_emit('EQ', left_reg, right_reg, 0)
            elif node_type == 'NotEqToOp':
                self.vm_emit('EQ', left_reg, right_reg, 1)  # Inverted
            elif node_type == 'LessThanOp':
                self.vm_emit('LT', left_reg, right_reg, 0)
            elif node_type == 'GreaterThanOp':
                self.vm_emit('LT', right_reg, left_reg, 0)  # Swap operands
            elif node_type == 'LessOrEqThanOp':
                self.vm_emit('LE', left_reg, right_reg, 0)
            elif node_type == 'GreaterOrEqThanOp':
                self.vm_emit('LE', right_reg, left_reg, 0)  # Swap operands
            
            # Result is in comparison flag, store to target
            self.vm_emit('LOADBOOL', target_reg, 1, 1)  # true, skip next
            self.vm_emit('LOADBOOL', target_reg, 0, 0)  # false
            
            return target_reg
        
        # Unary minus
        elif node_type == 'UMinusOp':
            operand_reg = self.compile_expression(node.operand)
            self.vm_emit('FAST_UNM', target_reg, operand_reg, 0)
            return target_reg
        
        # Logical not
        elif node_type == 'NotOp':
            operand_reg = self.compile_expression(node.operand)
            self.vm_emit('FAST_NOT', target_reg, operand_reg, 0)
            return target_reg
        
        # Length operator
        elif node_type == 'LengthOp':
            operand_reg = self.compile_expression(node.operand)
            self.vm_emit('FAST_LEN', target_reg, operand_reg, 0)
            return target_reg
        
        # String concatenation
        elif node_type == 'ConcatOp':
            left_reg = self.compile_expression(node.left)
            right_reg = self.compile_expression(node.right)
            self.vm_emit('CONCAT', target_reg, left_reg, right_reg)
            return target_reg
        
        # Table constructor
        elif node_type == 'Table':
            self.vm_emit('NEWTABLE', target_reg, 0, 0)
            
            # Compile table fields
            if hasattr(node, 'fields') and node.fields:
                for i, field in enumerate(node.fields):
                    field_type = type(field).__name__
                    
                    if field_type == 'Field':
                        # key = value
                        key_reg = self.compile_expression(field.key)
                        val_reg = self.compile_expression(field.value)
                        self.vm_emit('SETTABLE', target_reg, key_reg, val_reg)
                    else:
                        # Array-style: [i] = value
                        idx_const = self.vm_add_constant(i + 1)
                        idx_reg = self.vm_alloc_register()
                        self.vm_emit('LOADK', idx_reg, idx_const, 0)
                        val_reg = self.compile_expression(field)
                        self.vm_emit('SETTABLE', target_reg, idx_reg, val_reg)
            
            return target_reg
        
        # Index access: table[key] or table.key
        elif node_type == 'Index':
            table_reg = self.compile_expression(node.value)
            
            if hasattr(node, 'idx'):
                key_reg = self.compile_expression(node.idx)
            else:
                # Dot notation - key is a string
                key_const = self.vm_add_constant(node.idx.id if hasattr(node.idx, 'id') else str(node.idx))
                key_reg = self.vm_alloc_register()
                self.vm_emit('LOADK', key_reg, key_const, 0)
            
            self.vm_emit('GETTABLE', target_reg, table_reg, key_reg)
            return target_reg
        
        # Function call
        elif node_type == 'Call':
            func_reg = self.compile_expression(node.func)
            
            # Compile arguments
            arg_count = 0
            if hasattr(node, 'args') and node.args:
                for arg in node.args:
                    arg_reg = self.vm_alloc_register()
                    self.compile_expression(arg, arg_reg)
                    arg_count += 1
            
            # CALL: func_reg, arg_count, return_count
            self.vm_emit('CALL', func_reg, arg_count, 1)
            
            # Result is in func_reg
            if func_reg != target_reg:
                self.vm_emit('MOV', target_reg, func_reg, 0)
            
            return target_reg
        
        # Method call: obj:method(args)
        elif node_type == 'Invoke':
            # Get object
            obj_reg = self.compile_expression(node.source)
            
            # Get method name
            method_const = self.vm_add_constant(node.func.id if hasattr(node.func, 'id') else str(node.func))
            method_reg = self.vm_alloc_register()
            self.vm_emit('LOADK', method_reg, method_const, 0)
            
            # SELF: prepares obj and method
            self.vm_emit('SELF', target_reg, obj_reg, method_reg)
            
            # Compile arguments (obj is already at target_reg + 1)
            arg_count = 1  # Self counts as first arg
            if hasattr(node, 'args') and node.args:
                for arg in node.args:
                    arg_reg = self.vm_alloc_register()
                    self.compile_expression(arg, arg_reg)
                    arg_count += 1
            
            self.vm_emit('CALL', target_reg, arg_count, 1)
            return target_reg
        
        # Logical AND
        elif node_type == 'AndOp':
            left_reg = self.compile_expression(node.left, target_reg)
            
            # If left is false, skip right evaluation
            end_label = self.vm_create_label()
            self.vm_emit_jump('JMPIFNOT', end_label, left_reg)
            
            # Evaluate right side
            self.compile_expression(node.right, target_reg)
            
            self.vm_set_label(end_label)
            return target_reg
        
        # Logical OR
        elif node_type == 'OrOp':
            left_reg = self.compile_expression(node.left, target_reg)
            
            # If left is true, skip right evaluation
            end_label = self.vm_create_label()
            self.vm_emit_jump('JMPIF', end_label, left_reg)
            
            # Evaluate right side
            self.compile_expression(node.right, target_reg)
            
            self.vm_set_label(end_label)
            return target_reg
        
        # Anonymous function
        elif node_type in ('Function', 'AnonymousFunction'):
            # For now, emit a placeholder - full function compilation is complex
            # This would need to create a sub-bytecode chunk
            self.vm_emit('LOADNIL', target_reg, 0, 0)  # Placeholder
            return target_reg
        
        # Default: unknown node type
        else:
            # Try to handle as a generic node with a value
            self.vm_emit('LOADNIL', target_reg, 0, 0)
            return target_reg
    
    def compile_statement(self, node):
        """
        Compile a statement AST node to bytecode.
        
        Args:
            node: The AST statement node
        """
        node_type = type(node).__name__
        
        # Local variable assignment: local x = expr
        if node_type == 'LocalAssign':
            for i, target in enumerate(node.targets):
                if hasattr(target, 'id'):
                    var_name = target.id
                    reg = self.vm_alloc_register()
                    self.vm_var_registers[var_name] = reg
                    
                    if node.values and i < len(node.values):
                        self.compile_expression(node.values[i], reg)
                    else:
                        self.vm_emit('LOADNIL', reg, 0, 0)
        
        # Assignment: x = expr
        elif node_type == 'Assign':
            for i, target in enumerate(node.targets):
                target_type = type(target).__name__
                
                if target_type == 'Name':
                    var_name = target.id
                    
                    if var_name in self.vm_var_registers:
                        # Local variable
                        reg = self.vm_var_registers[var_name]
                        if node.values and i < len(node.values):
                            self.compile_expression(node.values[i], reg)
                    else:
                        # Global variable
                        val_reg = self.vm_alloc_register()
                        if node.values and i < len(node.values):
                            self.compile_expression(node.values[i], val_reg)
                        const_idx = self.vm_add_constant(var_name)
                        self.vm_emit('SETGLOBAL', val_reg, const_idx, 0)
                        self.vm_free_register()
                
                elif target_type == 'Index':
                    # Table assignment: t[k] = v
                    table_reg = self.compile_expression(target.value)
                    key_reg = self.compile_expression(target.idx)
                    val_reg = self.vm_alloc_register()
                    if node.values and i < len(node.values):
                        self.compile_expression(node.values[i], val_reg)
                    self.vm_emit('SETTABLE', table_reg, key_reg, val_reg)
        
        # Function call statement
        elif node_type == 'Call':
            self.compile_expression(node)
        
        # Method call statement
        elif node_type == 'Invoke':
            self.compile_expression(node)
        
        # If statement
        elif node_type == 'If':
            else_label = self.vm_create_label()
            end_label = self.vm_create_label()
            
            # Compile condition
            cond_reg = self.compile_expression(node.test)
            self.vm_emit_jump('JMPIFNOT', else_label, cond_reg)
            
            # Compile 'then' block
            self.vm_push_scope()
            if hasattr(node, 'body') and node.body:
                self.compile_block(node.body)
            self.vm_pop_scope()
            
            self.vm_emit_jump('JMP', end_label)
            
            # Compile 'else' block
            self.vm_set_label(else_label)
            if hasattr(node, 'orelse') and node.orelse:
                self.vm_push_scope()
                self.compile_block(node.orelse)
                self.vm_pop_scope()
            
            self.vm_set_label(end_label)
        
        # While loop
        elif node_type == 'While':
            loop_start = self.vm_create_label()
            loop_end = self.vm_create_label()
            
            self.vm_set_label(loop_start)
            
            # Compile condition
            cond_reg = self.compile_expression(node.test)
            self.vm_emit_jump('JMPIFNOT', loop_end, cond_reg)
            
            # Compile body
            self.vm_push_scope()
            if hasattr(node, 'body') and node.body:
                self.compile_block(node.body)
            self.vm_pop_scope()
            
            self.vm_emit_jump('JMP', loop_start)
            self.vm_set_label(loop_end)
        
        # Repeat-until loop
        elif node_type == 'Repeat':
            loop_start = self.vm_create_label()
            
            self.vm_set_label(loop_start)
            
            # Compile body
            self.vm_push_scope()
            if hasattr(node, 'body') and node.body:
                self.compile_block(node.body)
            
            # Compile condition
            cond_reg = self.compile_expression(node.test)
            self.vm_emit_jump('JMPIFNOT', loop_start, cond_reg)
            self.vm_pop_scope()
        
        # Numeric for loop
        elif node_type == 'Fornum':
            # for i = start, stop, step do body end
            loop_start = self.vm_create_label()
            loop_end = self.vm_create_label()
            
            self.vm_push_scope()
            
            # Allocate registers for loop control
            iter_reg = self.vm_alloc_register()
            limit_reg = self.vm_alloc_register()
            step_reg = self.vm_alloc_register()
            
            # Compile start, stop, step
            self.compile_expression(node.start, iter_reg)
            self.compile_expression(node.stop, limit_reg)
            if hasattr(node, 'step') and node.step:
                self.compile_expression(node.step, step_reg)
            else:
                # Default step = 1
                const_idx = self.vm_add_constant(1)
                self.vm_emit('LOADK', step_reg, const_idx, 0)
            
            # Bind loop variable
            if hasattr(node.target, 'id'):
                self.vm_var_registers[node.target.id] = iter_reg
            
            self.vm_set_label(loop_start)
            
            # Check condition: iter <= limit (for positive step)
            self.vm_emit('LE', iter_reg, limit_reg, 0)
            self.vm_emit_jump('JMPIFNOT', loop_end, iter_reg)
            
            # Compile body
            if hasattr(node, 'body') and node.body:
                self.compile_block(node.body)
            
            # Increment: iter = iter + step
            self.vm_emit('FAST_ADD', iter_reg, iter_reg, step_reg)
            self.vm_emit_jump('JMP', loop_start)
            
            self.vm_set_label(loop_end)
            self.vm_pop_scope()
        
        # Return statement
        elif node_type == 'Return':
            if hasattr(node, 'values') and node.values:
                ret_reg = self.compile_expression(node.values[0])
                self.vm_emit('RETURN', ret_reg, 1, 0)
            else:
                self.vm_emit('RETURN', 0, 0, 0)
        
        # Break statement
        elif node_type == 'Break':
            # Would need loop context to know where to jump
            # For now, emit a NOP
            self.vm_emit('NOP', 0, 0, 0)
        
        # Local function
        elif node_type == 'LocalFunction':
            # Placeholder - full function compilation is complex
            if hasattr(node.name, 'id'):
                reg = self.vm_alloc_register()
                self.vm_var_registers[node.name.id] = reg
                self.vm_emit('LOADNIL', reg, 0, 0)
        
        # Do block
        elif node_type == 'Do':
            self.vm_push_scope()
            if hasattr(node, 'body') and node.body:
                self.compile_block(node.body)
            self.vm_pop_scope()
        
        # Comment or other non-executable
        elif node_type in ('Comment', 'SemiColon'):
            pass  # Skip
        
        # Default: try to compile as expression
        else:
            try:
                self.compile_expression(node)
            except:
                pass  # Skip unknown statements
    
    def compile_block(self, block):
        """
        Compile a block of statements.
        
        Args:
            block: A Block node or list of statements
        """
        statements = block
        
        # Handle Block wrapper
        if hasattr(block, 'body'):
            statements = block.body
        
        if isinstance(statements, list):
            for stmt in statements:
                self.compile_statement(stmt)
        else:
            self.compile_statement(statements)
    
    def compile_ast(self, ast_node):
        """
        Compile an entire AST to bytecode.
        
        Args:
            ast_node: The root AST node (usually a Chunk)
            
        Returns:
            tuple: (bytecode_list, constants_list)
        """
        # Initialize virtualizer state
        self.init_virtualizer()
        
        # Compile the AST
        if hasattr(ast_node, 'body'):
            self.compile_block(ast_node.body)
        
        # Resolve jump labels
        self.vm_resolve_jumps()
        
        # Add final return if not present
        if not self.vm_bytecode or self.vm_bytecode[-1][0] != self.get_opcode('RETURN'):
            self.vm_emit('RETURN', 0, 0, 0)
        
        return (self.vm_bytecode, self.vm_constants_pool)
    
    def generate_vm_bytecode_table(self, bytecode):
        """
        Generate Lua code for the bytecode as a flat array.
        More efficient than nested tables for large bytecode.
        
        Args:
            bytecode: List of instruction tuples
            
        Returns:
            str: Lua table literal
        """
        # Flatten to 1D array: {op1, a1, b1, c1, op2, a2, b2, c2, ...}
        flat = []
        for instr in bytecode:
            flat.extend(instr)
        
        return "{" + ",".join(str(x) for x in flat) + "}"
    
    def generate_vm_constants_table(self, constants):
        """
        Generate Lua code for the VM constants pool.
        
        Args:
            constants: List of constant values
            
        Returns:
            str: Lua table literal
        """
        items = []
        for const in constants:
            if isinstance(const, str):
                # Escape string
                escaped = const.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')
                items.append(f'"{escaped}"')
            elif isinstance(const, bool):
                items.append('true' if const else 'false')
            elif const is None:
                items.append('nil')
            else:
                items.append(str(const))
        
        return "{" + ",".join(items) + "}"
    
    # =========================================================================
    # HOT PATH DETECTION - Selective Virtualization
    # =========================================================================
    
    def is_hot_path(self, node):
        """
        Determine if an AST node is a "hot path" that should remain native.
        
        Hot paths include:
        - while true do loops (main game loops)
        - Functions named Step, Update, Render
        - High-frequency polling (iskeypressed, etc.)
        - Drawing property updates
        
        Args:
            node: AST node to check
            
        Returns:
            bool: True if this should remain native code
        """
        node_type = type(node).__name__
        
        # Check for infinite loops: while true do
        if node_type == 'While':
            if hasattr(node, 'test'):
                test_type = type(node.test).__name__
                if test_type == 'TrueExpr':
                    return True
        
        # Check for Step/Update/Render functions
        if node_type in ('Function', 'LocalFunction'):
            if hasattr(node, 'name'):
                name = node.name
                if hasattr(name, 'id'):
                    func_name = name.id.lower()
                    if any(hot in func_name for hot in ['step', 'update', 'render', 'draw', 'loop']):
                        return True
        
        # Check for method definitions on UILib/Window
        if node_type == 'Method':
            if hasattr(node, 'name'):
                method_name = str(node.name).lower()
                if 'step' in method_name:
                    return True
        
        return False
    
    def detect_hot_paths(self, node, hot_paths=None):
        """
        Walk the AST and identify all hot paths.
        
        Args:
            node: Root AST node
            hot_paths: List to collect hot path nodes (created if None)
            
        Returns:
            list: List of AST nodes that are hot paths
        """
        if hot_paths is None:
            hot_paths = []
        
        if node is None:
            return hot_paths
        
        if self.is_hot_path(node):
            hot_paths.append(node)
        
        # Recurse into children
        for attr_name, attr_value in node.__dict__.items():
            if attr_name.startswith('_'):
                continue
            
            if isinstance(attr_value, list):
                for item in attr_value:
                    if hasattr(item, '__dict__'):
                        self.detect_hot_paths(item, hot_paths)
            elif hasattr(attr_value, '__dict__'):
                self.detect_hot_paths(attr_value, hot_paths)
        
        return hot_paths
    
    def split_ast_for_virtualization(self, ast_node):
        """
        Split the AST into virtualized and native sections.
        
        Args:
            ast_node: Root AST node
            
        Returns:
            tuple: (statements_to_virtualize, hot_path_statements)
        """
        # Get the statements list
        if hasattr(ast_node, 'body'):
            body = ast_node.body
            if hasattr(body, 'body'):
                statements = body.body
            else:
                statements = body if isinstance(body, list) else [body]
        else:
            statements = [ast_node]
        
        virtualize = []
        native = []
        
        for stmt in statements:
            if self.is_hot_path(stmt):
                native.append(stmt)
            else:
                virtualize.append(stmt)
        
        return (virtualize, native)
    
    def generate_hybrid_output(self, virtualized_bytecode, virtualized_constants, native_statements):
        """
        Generate the hybrid output with both VM code and native code.
        
        Args:
            virtualized_bytecode: Bytecode for virtualized sections
            virtualized_constants: Constants for virtualized sections
            native_statements: AST nodes to output as native Lua
            
        Returns:
            str: Combined Lua code
        """
        lines = []
        
        # Generate VM bytecode table
        bc_name = self.generate_var_name()
        bc_lua = self.generate_vm_bytecode_table(virtualized_bytecode)
        lines.append(f"local {bc_name} = {bc_lua}")
        
        # Generate VM constants table  
        k_name = self.generate_var_name()
        k_lua = self.generate_vm_constants_table(virtualized_constants)
        lines.append(f"local {k_name} = {k_lua}")
        
        # Generate VM executor
        lines.append(self.generate_vm_executor())
        
        # Execute virtualized setup code
        lines.append(f"local _vm_result = {self.vm_execute_name}({bc_name}, {k_name}, nil)")
        
        # Output native hot paths as regular Lua
        if native_statements:
            lines.append("-- Native hot paths (unvirtualized for performance)")
            for stmt in native_statements:
                try:
                    stmt_lua = ast.to_lua_source(stmt)
                    lines.append(stmt_lua)
                except:
                    pass  # Skip if conversion fails
        
        return "\n".join(lines)
    
    def generate_mutated_expression(self, value, depth=0):
        """
        Turns a simple integer into an obfuscated math expression.
        
        Uses only simple arithmetic with literal numbers to guarantee correct results.
        No recursion to avoid operator precedence issues.
        
        Args:
            value (int): The target number (must be positive for array indices).
            
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
        
        # Choose a random strategy - all use only Number() nodes to avoid nesting issues
        strategy = random.randint(1, 4)
        
        if strategy == 1 and value > 1:
            # Simple split: value = a + b
            a = random.randint(1, value - 1)
            b = value - a
            if SPECIFIC_BINOP_MODE:
                return ADD_OP(Number(a), Number(b))
            else:
                return BINOP(ADD_OP(), Number(a), Number(b))
        
        elif strategy == 2 and value > 2:
            # Triple split: value = a + b + c (all positive)
            a = random.randint(1, max(1, value // 3))
            remainder = value - a
            b = random.randint(1, max(1, remainder - 1))
            c = remainder - b
            
            if c < 1:
                # Fallback to simple
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
            # Subtraction: value = (value + offset) - offset
            offset = random.randint(50, 300)
            larger = value + offset
            if SPECIFIC_BINOP_MODE:
                return SUB_OP(Number(larger), Number(offset))
            else:
                return BINOP(SUB_OP(), Number(larger), Number(offset))
        
        else:
            # Multi-term with subtraction: a + b + c - d = value
            # Ensure (a + b + c) > d and result = value
            d = random.randint(50, 200)
            total = value + d  # a + b + c must equal this
            a = random.randint(1, max(1, total // 3))
            remainder = total - a
            b = random.randint(1, max(1, remainder // 2))
            c = remainder - b
            
            if a < 1 or b < 1 or c < 1:
                # Fallback to simple
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

    # =========================================================================
    # CLOSURE-BASED VIRTUALIZATION SYSTEM
    # =========================================================================
    
    def generate_virt_key(self, op_type):
        """
        Generate a unique random key for a virtualized operation.
        
        Args:
            op_type (str): The operation type (e.g., 'add', 'sub', 'mul', 'div', 'eq', 'neq', 'lt', 'gt', 'le', 'ge')
            
        Returns:
            int: A unique random key for this operation
        """
        # Generate a unique key that hasn't been used
        while True:
            key = random.randint(100, 9999)
            if key not in self.virt_ops.values():
                self.virt_ops[op_type] = key
                return key
    
    def get_op_type_name(self, node):
        """
        Get the operation type string from an AST node.
        
        Args:
            node: An AST binary operation node
            
        Returns:
            str or None: The operation type name, or None if not recognized
        """
        node_type = type(node).__name__
        
        # Map node type names to our operation names
        op_map = {
            'AddOp': 'add',
            'SubOp': 'sub', 
            'MultOp': 'mul',
            'DivOp': 'div',
            'ModOp': 'mod',
            'EqToOp': 'eq',
            'NotEqToOp': 'neq',
            'LessThanOp': 'lt',
            'GreaterThanOp': 'gt',
            'LessOrEqThanOp': 'le',
            'GreaterOrEqThanOp': 'ge',
            'AndOp': 'and_op',
            'OrOp': 'or_op',
            'ConcatOp': 'concat',
        }
        
        return op_map.get(node_type, None)
    
    def generate_virtualized_ops_table(self):
        """
        Generate the Lua code for the virtualized operators table.
        
        This creates a table of anonymous functions, each performing a basic
        operation. Uses clean arithmetic to support Vectors and other userdata types.
        
        Returns:
            str: Lua code defining the operators table
        """
        lines = []
        lines.append(f"local {self.virt_table_name} = {{}}")
        
        # Generate functions for each registered operation
        # Using clean arithmetic (no modifiers) to support Vectors/userdata
        for op_type, key in self.virt_ops.items():
            
            if op_type == 'add':
                # Clean addition - supports Numbers, Vectors, etc.
                lines.append(f"{self.virt_table_name}[{key}] = function(a, b) return a + b end")
                
            elif op_type == 'sub':
                # Clean subtraction - supports Numbers, Vectors, etc.
                lines.append(f"{self.virt_table_name}[{key}] = function(a, b) return a - b end")
                
            elif op_type == 'mul':
                # Clean multiplication - supports Numbers, Vectors, etc.
                lines.append(f"{self.virt_table_name}[{key}] = function(a, b) return a * b end")
                
            elif op_type == 'div':
                # Clean division - supports Numbers, Vectors, etc.
                lines.append(f"{self.virt_table_name}[{key}] = function(a, b) return a / b end")
                
            elif op_type == 'mod':
                # Modulo
                lines.append(f"{self.virt_table_name}[{key}] = function(a, b) return a % b end")
                
            elif op_type == 'eq':
                # Equality
                lines.append(f"{self.virt_table_name}[{key}] = function(a, b) return a == b end")
                
            elif op_type == 'neq':
                # Not equal
                lines.append(f"{self.virt_table_name}[{key}] = function(a, b) return a ~= b end")
                
            elif op_type == 'lt':
                # Less than
                lines.append(f"{self.virt_table_name}[{key}] = function(a, b) return a < b end")
                
            elif op_type == 'gt':
                # Greater than
                lines.append(f"{self.virt_table_name}[{key}] = function(a, b) return a > b end")
                
            elif op_type == 'le':
                # Less or equal
                lines.append(f"{self.virt_table_name}[{key}] = function(a, b) return a <= b end")
                
            elif op_type == 'ge':
                # Greater or equal
                lines.append(f"{self.virt_table_name}[{key}] = function(a, b) return a >= b end")
                
            elif op_type == 'and_op':
                # Logical and
                lines.append(f"{self.virt_table_name}[{key}] = function(a, b) return a and b end")
                
            elif op_type == 'or_op':
                # Logical or
                lines.append(f"{self.virt_table_name}[{key}] = function(a, b) return a or b end")
                
            elif op_type == 'concat':
                # String concatenation
                lines.append(f"{self.virt_table_name}[{key}] = function(a, b) return a .. b end")
        
        return "\n".join(lines) + "\n"
    
    def create_virt_call_node(self, op_key, left_node, right_node):
        """
        Create an AST node for a virtualized operator call.
        
        Transforms: a + b  ->  _VT[key](a, b)
        
        Args:
            op_key (int): The key in the virtualized ops table
            left_node: The left operand AST node
            right_node: The right operand AST node
            
        Returns:
            Call: An AST Call node representing _VT[key](a, b)
        """
        # Create _VT[key]
        table_node = Name(self.virt_table_name)
        key_node = Number(op_key)
        func_index = Index(key_node, table_node, notation=1)
        
        # Create the call with left and right as arguments
        return Call(func_index, [left_node, right_node])
    
    def virtualize_operations(self, node):
        """
        Traverse the AST and replace binary operations with virtualized function calls.
        
        This transforms expressions like `a + b` into `_VT[key](a, b)` where _VT is
        a table of operator functions.
        
        Args:
            node: The AST node to process
        """
        if node is None:
            return
        
        # List of binary operation node type names to virtualize
        binary_ops = {
            'AddOp', 'SubOp', 'MultOp', 'DivOp', 'ModOp',
            'EqToOp', 'NotEqToOp', 'LessThanOp', 'GreaterThanOp',
            'LessOrEqThanOp', 'GreaterOrEqThanOp',
            'AndOp', 'OrOp', 'ConcatOp'
        }
        
        # Iterate over all attributes
        for attr_name, attr_value in list(node.__dict__.items()):
            if attr_name.startswith('_'):
                continue
            
            if isinstance(attr_value, list):
                for i, item in enumerate(attr_value):
                    if isinstance(item, Node):
                        node_type = type(item).__name__
                        
                        # Check if this is a binary operation we should virtualize
                        if node_type in binary_ops and hasattr(item, 'left') and hasattr(item, 'right'):
                            op_type = self.get_op_type_name(item)
                            if op_type:
                                # Get or create a key for this operation type
                                if op_type not in self.virt_ops:
                                    self.generate_virt_key(op_type)
                                
                                op_key = self.virt_ops[op_type]
                                
                                # First, recursively process the operands
                                self.virtualize_operations(item.left)
                                self.virtualize_operations(item.right)
                                
                                # Create the virtualized call
                                virt_call = self.create_virt_call_node(op_key, item.left, item.right)
                                
                                # Replace the operation with the call
                                attr_value[i] = virt_call
                        else:
                            # Recurse into child nodes
                            self.virtualize_operations(item)
            
            elif isinstance(attr_value, Node):
                node_type = type(attr_value).__name__
                
                # Check if this is a binary operation we should virtualize
                if node_type in binary_ops and hasattr(attr_value, 'left') and hasattr(attr_value, 'right'):
                    op_type = self.get_op_type_name(attr_value)
                    if op_type:
                        # Get or create a key for this operation type
                        if op_type not in self.virt_ops:
                            self.generate_virt_key(op_type)
                        
                        op_key = self.virt_ops[op_type]
                        
                        # First, recursively process the operands
                        self.virtualize_operations(attr_value.left)
                        self.virtualize_operations(attr_value.right)
                        
                        # Create the virtualized call
                        virt_call = self.create_virt_call_node(op_key, attr_value.left, attr_value.right)
                        
                        # Replace the operation with the call
                        setattr(node, attr_name, virt_call)
                else:
                    # Recurse into child nodes
                    self.virtualize_operations(attr_value)

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
        Encrypts a string into a list of XOR-ed integers using dynamic runtime key.
        
        The key is computed as: (base + (runtime_component * mult + add) + index) % 256
        At runtime, runtime_component comes from Matcha-specific globals.
        For encryption, we use a known value (game string length) that will match at runtime.
        
        Args:
            text (str): The raw string to encrypt.
            
        Returns:
            list: A list of integers where each byte is XORed with the dynamic key.
        """
        encrypted = []
        # Runtime component simulation - uses len("game") which is 4 in Matcha
        # This value is computed at runtime from tostring(game):len() % 10
        runtime_component = 4  # len("game") = 4
        
        for i, char in enumerate(text):
            # Dynamic key formula matching the runtime decryptor
            k = (self.xor_key_base + (runtime_component * self.xor_key_runtime_mult + self.xor_key_runtime_add) + i) % 256
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
        
        # Generate randomized names for bit32 functions (stealth mode)
        bit32_table_name = self.generate_var_name()  # Instead of 'bit32'
        bit32_bnot_name = self.generate_var_name()
        bit32_band_name = self.generate_var_name()
        bit32_bor_name = self.generate_var_name()
        bit32_bxor_name = self.generate_var_name()
        bit32_lshift_name = self.generate_var_name()
        bit32_rshift_name = self.generate_var_name()
        bit32_arshift_name = self.generate_var_name()
        
        # Randomized internal variable names
        n_var = self.generate_var_name()  # For N = 32
        p_var = self.generate_var_name()  # For P = 2^N
        
        # Define stealthy Bit32 implementation wrapped in local scope
        # Uses randomized names, no global pollution, invisible to getgenv()/_G
        bit32_polyfill = f"""local {bit32_table_name}
do
local {n_var} = 32
local {p_var} = 2 ^ {n_var}
local {bit32_bnot_name} = function(x)
x = x % {p_var}
return ({p_var} - 1) - x
end
local {bit32_band_name} = function(x, y)
if (y == 255) then return x % 256 end
if (y == 65535) then return x % 65536 end
if (y == 4294967295) then return x % 4294967296 end
x, y = x % {p_var}, y % {p_var}
local r = 0
local p = 1
for i = 1, {n_var} do
local a, b = x % 2, y % 2
x, y = math.floor(x / 2), math.floor(y / 2)
if ((a + b) == 2) then r = r + p end
p = 2 * p
end
return r
end
local {bit32_bor_name} = function(x, y)
if (y == 255) then return (x - (x % 256)) + 255 end
if (y == 65535) then return (x - (x % 65536)) + 65535 end
if (y == 4294967295) then return 4294967295 end
x, y = x % {p_var}, y % {p_var}
local r = 0
local p = 1
for i = 1, {n_var} do
local a, b = x % 2, y % 2
x, y = math.floor(x / 2), math.floor(y / 2)
if ((a + b) >= 1) then r = r + p end
p = 2 * p
end
return r
end
local {bit32_bxor_name} = function(x, y)
x, y = x % {p_var}, y % {p_var}
local r = 0
local p = 1
for i = 1, {n_var} do
local a, b = x % 2, y % 2
x, y = math.floor(x / 2), math.floor(y / 2)
if ((a + b) == 1) then r = r + p end
p = 2 * p
end
return r
end
local {bit32_lshift_name} = function(x, s)
if (math.abs(s) >= {n_var}) then return 0 end
x = x % {p_var}
if (s < 0) then return math.floor(x * (2 ^ s))
else return (x * (2 ^ s)) % {p_var} end
end
local {bit32_rshift_name} = function(x, s)
if (math.abs(s) >= {n_var}) then return 0 end
x = x % {p_var}
if (s > 0) then return math.floor(x * (2 ^ -s))
else return (x * (2 ^ -s)) % {p_var} end
end
local {bit32_arshift_name} = function(x, s)
if (math.abs(s) >= {n_var}) then return 0 end
x = x % {p_var}
if (s > 0) then
local add = 0
if (x >= ({p_var} / 2)) then add = {p_var} - (2 ^ ({n_var} - s)) end
return math.floor(x * (2 ^ -s)) + add
else return (x * (2 ^ -s)) % {p_var} end
end
{bit32_table_name} = {{['{bit32_bnot_name}']={bit32_bnot_name},['{bit32_band_name}']={bit32_band_name},['{bit32_bor_name}']={bit32_bor_name},['{bit32_bxor_name}']={bit32_bxor_name},['{bit32_lshift_name}']={bit32_lshift_name},['{bit32_rshift_name}']={bit32_rshift_name},['{bit32_arshift_name}']={bit32_arshift_name}}}
end
"""

        # Generate additional randomized names for runtime key computation
        runtime_key_var = self.generate_var_name()
        game_check_var = self.generate_var_name()
        
        # Define the Lua decryptor function string with DYNAMIC RUNTIME KEY
        # The key is computed using Matcha-specific globals that only exist at runtime
        # If run in a dumper/different environment, strings decrypt to garbage
        decryptor_lua = (
            f"local {runtime_key_var}\n"
            f"do\n"
            f"    local {game_check_var} = tostring(game):len() % 10\n"
            f"    {runtime_key_var} = {self.xor_key_base} + ({game_check_var} * {self.xor_key_runtime_mult} + {self.xor_key_runtime_add})\n"
            f"end\n"
            f"local function {decryptor_name}(bytes)\n"
            f"    local res = {{}}\n"
            f"    for i, b in ipairs(bytes) do\n"
            f"        local k = ({runtime_key_var} + (i - 1)) % 256\n"
            f"        table.insert(res, string.char({bit32_table_name}['{bit32_bxor_name}'](b, k)))\n"
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
        
        # 2. Generate Virtualized Operators Table (if any operations were virtualized)
        virt_ops_lua = ""
        if self.virt_ops:
            virt_ops_lua = self.generate_virtualized_ops_table()
        
        # 3. Generate the Script Body
        # Use ast.to_lua_source(self.ast) to convert modified tree back to Lua source
        script_lua = ast.to_lua_source(self.ast)
        
        # 4. Combine and Write
        # Order: Polyfill -> Decryptor -> Constant Pool -> Virt Ops -> Script
        final_output = bit32_polyfill + "\n" + decryptor_lua + pool_lua + "\n\n" + virt_ops_lua + "\n" + script_lua
        
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

    # 10. Virtualize Binary Operations
    print("[*] Virtualizing Binary Operations...")
    obfuscator.virtualize_operations(obfuscator.ast)
    print(f"[+] Operations Virtualized: {len(obfuscator.virt_ops)} operator types")

    # 11. Flatten Root Flow
    print("[*] Flattening Root Control Flow...")
    obfuscator.flatten_root_flow(obfuscator.ast)
    print("[+] Root Control Flow Flattened")
    
    # 12. Generate Output
    print("[*] Generating output file...")
    obfuscator.generate_output()
    print(f"[+] Obfuscated script written to {obfuscator.output_file}")
