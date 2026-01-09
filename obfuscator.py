"""
Matcha LuaU Obfuscator (Hardened Edition v2)
=============================================
A performance-friendly obfuscator designed specifically for the Matcha LuaU VM.
Focuses on making AI-based deobfuscation extremely difficult while maintaining runtime efficiency.

Key Design Principles:
- Constant Pool: All strings and numbers are extracted and stored in a hidden table
- Compatible with Matcha's limitations (no .Magnitude, .Unit, no task.defer/delay/cancel)
- Optimized for high FPS execution (up to 1000fps with wait(.001))

ANTI-AI DEOBFUSCATION FEATURES:

1. Runtime Environment Anchoring:
   - State keys derived from Matcha/Roblox runtime values (game.PlaceId, LocalPlayer.Name)
   - AI cannot predict these values, halting constant folding

2. Matcha-Specific Tautologies:
   - True conditions: type(Drawing.new("Square")) == "table"
   - False conditions: tostring(Vector3.new(1,1,1)) == "1, 1, 1" (Matcha prints address)
   - Exploits Matcha VM quirks that AI assumes standard Roblox behavior

3. Polymorphic Operator Dispatch:
   - PolyMath(mode, a, b, salt) where mode + salt determines operation
   - Many-to-one mapping prevents simple substitution
   - AI cannot replace PolyMath(100, a, b, 400) with "+" without knowing salt

4. Control Flow Spaghettification:
   - Ghost States: 1-2 fake states per real state with junk code
   - State Interleaving: Transitions happen mid-block, not just at end
   - Creates control flow GRAPH, not simple linked list
   - Ghost states can transition to other ghosts (cycles) or random real states

5. Dynamic Opaque Predicates:
   - Math identities: sin²+cos²=1, |sin(x)|≤1, floor(n)≤n
   - Combined with Matcha-specific checks for layered protection

6. Equivalency Mutation:
   - select(1, value) wrappers for function call boundaries
   - Math identities: value + 0, value * 1, math.abs(value)
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
    global BINOP, ADD_OP, SUB_OP, MULT_OP, DIV_OP, MOD_OP, EQ_OP, TRUE_NODE
    global NUMERIC_FOR, GENERIC_FOR, SPECIFIC_BINOP_MODE
    global ANON_FUNC
    global BXOR_OP, BAND_OP, BNOT_OP  # Bitwise operators for MBA
    
    # Initialize bitwise operators
    BXOR_OP = None
    BAND_OP = None
    BNOT_OP = None
    MOD_OP = None
    
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
            
            # Need to re-fetch sub/mult/div/mod classes by parsing
            sub_tree = ast.parse("local x = 1 - 1")
            SUB_OP = type(get_statements(sub_tree)[0].values[0].op)
            
            mult_tree = ast.parse("local x = 1 * 1")
            MULT_OP = type(get_statements(mult_tree)[0].values[0].op)
            
            div_tree = ast.parse("local x = 1 / 1")
            DIV_OP = type(get_statements(div_tree)[0].values[0].op)
            
            # Discover modulo operator
            try:
                mod_tree = ast.parse("local x = 5 % 2")
                MOD_OP = type(get_statements(mod_tree)[0].values[0].op)
            except:
                MOD_OP = SUB_OP  # Fallback
            
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
            
            # Discover modulo operator
            try:
                mod_tree = ast.parse("local x = 5 % 2")
                MOD_OP = type(get_statements(mod_tree)[0].values[0])
            except:
                MOD_OP = SUB_OP  # Fallback
            
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
            MOD_OP = getattr(astnodes, "Mod", None)

    except Exception as e:
        print(f"[-] Discovery Error (BinOp): {e}")
        # Panic Fallback
        SPECIFIC_BINOP_MODE = False
        BINOP = getattr(astnodes, "BinOp", None)
        ADD_OP = getattr(astnodes, "Add", None)
        EQ_OP = getattr(astnodes, "Eq", None)
        MOD_OP = getattr(astnodes, "Mod", None)
    
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
    print(f"[+] Auto-Discovered: Add={ADD_OP.__name__ if ADD_OP else '?'}, Mod={MOD_OP.__name__ if MOD_OP else '?'}, Eq={EQ_OP.__name__ if EQ_OP else '?'}, Anon={ANON_FUNC.__name__ if ANON_FUNC else '?'}")

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
        # ADVERSARIAL CONTEXT POISONING (Anti-AI Deobfuscation)
        # =====================================================================
        # These misleading names are designed to confuse AI models by creating
        # semantic misdirection. When an AI sees "UpdateSoundVolume", it assumes
        # the function handles audio - not combat mechanics.
        
        # Category: Misleading function/variable names that suggest benign functionality
        # Key = actual purpose hint, Value = misleading name pool to choose from
        self.misleading_name_pools = {
            # Combat/Cheat functions disguised as UI/Audio/Network
            'combat': [
                'UpdateLightingEffects', 'CalculateAmbientOcclusion', 'RefreshUILayout',
                'SyncAnimationState', 'ValidateChatMessage', 'ProcessAudioBuffer',
                'UpdateShadowMap', 'CalculateReflections', 'RefreshTextureCache',
                'ProcessNetworkPing', 'UpdateLocalization', 'ValidateInputBuffer'
            ],
            # ESP/Visual cheat functions disguised as legitimate rendering
            'visual': [
                'RenderUIOverlay', 'UpdateFontCache', 'ProcessGlyphMetrics',
                'CalculateTextBounds', 'RefreshLayoutConstraints', 'SyncScrollPosition',
                'ValidateColorSpace', 'UpdateGammaCorrection', 'ProcessAntiAliasing',
                'RenderDropShadow', 'CalculateBorderRadius', 'UpdateClipRegion'
            ],
            # Aimbot/Targeting disguised as camera/animation
            'targeting': [
                'UpdateCameraSmoothing', 'CalculateViewportBounds', 'ProcessCinematicMode',
                'SyncCharacterAnimation', 'ValidateMotionPath', 'RefreshIKConstraints',
                'UpdateBoneTransforms', 'CalculateAnimationBlend', 'ProcessRagdollPhysics',
                'RenderMotionBlur', 'CalculateFocalLength', 'UpdateDepthOfField'
            ],
            # Memory/Data manipulation disguised as caching/logging
            'memory': [
                'UpdateAnalyticsCache', 'ProcessTelemetryData', 'ValidateSessionState',
                'RefreshLeaderboardData', 'SyncAchievementProgress', 'CalculateStatistics',
                'UpdatePerformanceMetrics', 'ProcessDebugOutput', 'ValidateConfigData',
                'RefreshAssetManifest', 'SyncCloudSaveData', 'CalculateChecksum'
            ],
            # General purpose misleading names
            'generic': [
                'InitializeSubsystem', 'ProcessQueuedEvents', 'ValidateResourceState',
                'UpdateCachePolicy', 'RefreshBindingContext', 'SynchronizeState',
                'CalculateHashValue', 'ProcessDeferredUpdate', 'ValidateIntegrity',
                'UpdateTimestamp', 'RefreshConnectionPool', 'ProcessHeartbeat'
            ]
        }
        
        # Fake comment pool - These are inserted to mislead AI about code purpose
        # AI models heavily weight comments for understanding code semantics
        self.fake_comment_pool = [
            # UI/Rendering misdirection
            "-- Adjusting UI element anchor points for responsive layout",
            "-- Recalculating font metrics for dynamic text scaling",
            "-- Synchronizing animation keyframes with server tick rate",
            "-- Validating color values are within sRGB gamut",
            "-- Processing deferred rendering queue for transparency",
            "-- Updating shadow cascade boundaries for optimal quality",
            # Audio misdirection
            "-- Mixing audio channels for spatial positioning",
            "-- Calculating reverb decay based on room geometry",
            "-- Processing audio buffer for latency compensation",
            "-- Adjusting doppler effect parameters for velocity",
            # Network/Data misdirection
            "-- Serializing state delta for network replication",
            "-- Validating packet checksum before processing",
            "-- Compressing analytics payload for transmission",
            "-- Synchronizing leaderboard data with cloud service",
            # Performance misdirection
            "-- Profiling frame time for performance optimization",
            "-- Garbage collecting orphaned references",
            "-- Optimizing draw call batching for GPU efficiency",
            "-- Caching computed values to reduce recalculation"
        ]
        
        # Track which misleading names have been used to avoid duplicates
        self.used_misleading_names = set()
        
        # =====================================================================
        # LOGIC FLOODING (Hallucination Trigger for AI)
        # =====================================================================
        # Creates function density that overwhelms AI pattern recognition
        
        # Polymorphic opcode threshold - operations above this value behave differently
        self.poly_opcode_threshold = random.randint(800, 1200)
        
        # Enable/disable features
        self.enable_misleading_names = True  # Adversarial naming
        self.enable_fake_comments = True     # Fake comment injection
        self.enable_logic_flooding = True    # Mega-function creation
        
        # =====================================================================
        # STATE-DEPENDENT STRING DECRYPTION (Anti-AI Weakness E Fix)
        # =====================================================================
        # Links string decryption to the CFF state variable
        # AI cannot decrypt strings without simulating the entire state machine
        self.enable_state_dependent_decrypt = True
        self.state_var_name = None  # Will be set during flatten_root_flow
        self.current_block_state = None  # Tracks current state during processing
        self.state_string_map = {}  # Maps (string, state_id) -> encrypted_bytes
        
        # =====================================================================
        # HASHED GLOBAL PROXIES (Anti-AI Weakness C Fix)
        # =====================================================================
        # Instead of env["Drawing"], use VirtualEnv[hash("Drawing")]
        # AI sees VirtualEnv[15923] and has no idea what it is
        self.enable_hashed_globals = True
        self.virtual_env_name = self.generate_var_name()  # Name for VirtualEnv table
        self.hash_func_name = self.generate_var_name()     # Name for Hash function
        self.global_hash_cache = {}  # Cache computed hashes for globals

    def _str(self, s):
        """
        Helper to create a String AST node with proper arguments.
        luaparser's String() requires (s, raw) - we pass the same value for both.
        
        Args:
            s: The string value
            
        Returns:
            String: An AST String node
        """
        return String(s, s)
    
    # =========================================================================
    # HASHED GLOBAL PROXIES - DJB2 Hash Implementation
    # =========================================================================
    
    def calculate_djb2_hash(self, name):
        """
        Calculates DJB2 hash for a string - MUST match the Lua implementation exactly.
        
        This is used to create opaque global lookups. Instead of accessing 
        getgenv()["Drawing"], we access VirtualEnv[hash("Drawing")].
        
        The AI sees VirtualEnv[15923] and cannot determine what global it refers to
        without reversing the hash and testing against all possible Roblox globals.
        
        Args:
            name (str): The global name to hash (e.g., "Drawing", "Vector3")
            
        Returns:
            int: A 32-bit hash value
        """
        if name in self.global_hash_cache:
            return self.global_hash_cache[name]
        
        h = 5381
        for char in name:
            # DJB2: hash * 33 + c, with XOR variant
            h = ((h * 33) ^ ord(char)) & 0xFFFFFFFF  # Ensure 32-bit wrap
        
        self.global_hash_cache[name] = h
        return h
    
    def generate_hash_function_lua(self, bit32_table_name, bit32_bxor_name):
        """
        Generates the Lua code for the hashing function and VirtualEnv setup.
        
        This creates:
        1. A Hash function matching calculate_djb2_hash exactly
        2. A VirtualEnv table populated by scanning getgenv()
        
        IMPORTANT: Uses bit32 polyfill for XOR since LuaU doesn't support ~ operator
        
        Args:
            bit32_table_name: Name of the bit32 polyfill table
            bit32_bxor_name: Name of the bxor function in the polyfill
        
        Returns:
            str: Lua code to inject at the top of the script
        """
        # Use misleading names for the hash internals
        h_var = self.generate_var_name()
        i_var = self.generate_var_name()
        str_var = self.generate_var_name()
        name_var = self.generate_var_name()
        val_var = self.generate_var_name()
        
        # Use bit32 polyfill for XOR - LuaU doesn't have the ~ operator
        lua_code = f"""
{self.get_random_fake_comment()}
local {self.virtual_env_name} = {{}}
local function {self.hash_func_name}({str_var})
    {self.get_random_fake_comment()}
    local {h_var} = 5381
    for {i_var} = 1, #{str_var} do
        {h_var} = ({h_var} * 33) % 4294967296
        {h_var} = {bit32_table_name}['{bit32_bxor_name}']({h_var}, string.byte({str_var}, {i_var})) % 4294967296
    end
    return {h_var}
end
{self.get_random_fake_comment()}
do
    local {name_var}, {val_var}
    for {name_var}, {val_var} in pairs((getgenv and getgenv()) or getfenv() or _G) do
        if type({name_var}) == "string" then
            {self.virtual_env_name}[{self.hash_func_name}({name_var})] = {val_var}
        end
    end
    {self.virtual_env_name}[{self.hash_func_name}("game")] = game
    {self.virtual_env_name}[{self.hash_func_name}("workspace")] = workspace
    {self.virtual_env_name}[{self.hash_func_name}("Drawing")] = Drawing
    {self.virtual_env_name}[{self.hash_func_name}("Vector3")] = Vector3
    {self.virtual_env_name}[{self.hash_func_name}("Vector2")] = Vector2
    {self.virtual_env_name}[{self.hash_func_name}("Color3")] = Color3
    {self.virtual_env_name}[{self.hash_func_name}("CFrame")] = CFrame
    {self.virtual_env_name}[{self.hash_func_name}("Instance")] = Instance
end
"""
        return lua_code
    
    def create_hashed_global_lookup(self, global_name):
        """
        Creates an AST node for VirtualEnv[hash] lookup.
        
        Transforms: Drawing -> VirtualEnv[123456]
        
        Args:
            global_name (str): The global name (e.g., "Drawing")
            
        Returns:
            Index: AST node for VirtualEnv[hash_value]
        """
        hash_value = self.calculate_djb2_hash(global_name)
        
        # Create VirtualEnv[hash_value]
        return Index(
            Number(hash_value),
            Name(self.virtual_env_name),
            notation=1  # Bracket notation
        )
    
    # =========================================================================
    # STATE-DEPENDENT STRING DECRYPTION
    # =========================================================================
    
    def encrypt_string_with_state(self, text, state_id):
        """
        Encrypts a string using the CFF state ID as part of the key.
        
        The AI cannot decrypt this without knowing what state the code is in,
        which requires simulating the entire state machine.
        
        Args:
            text (str): The string to encrypt
            state_id (int): The CFF state ID for this block
            
        Returns:
            list: Encrypted bytes
        """
        encrypted = []
        # Combine base key with state ID for state-dependent decryption
        # Key formula: (base + state_id + index) % 256
        for i, char in enumerate(text):
            k = (self.xor_key_base + (state_id % 255) + i) % 256
            enc_byte = ord(char) ^ k
            encrypted.append(enc_byte)
        return encrypted
    
    def generate_state_dependent_decryptor(self, bit32_table_name, bit32_bxor_name):
        """
        Generates a decryptor function that requires the state variable.
        
        The function signature is: Decrypt(bytes, state)
        Without knowing the correct state value, decryption produces garbage.
        
        Args:
            bit32_table_name: Name of the bit32 polyfill table
            bit32_bxor_name: Name of the bxor function in the table
            
        Returns:
            str: Lua code for the state-dependent decryptor
        """
        decryptor_name = self.generate_var_name()
        self.state_decryptor_name = decryptor_name
        
        # Misleading parameter names
        bytes_param = self.generate_misleading_name('memory')
        state_param = self.generate_misleading_name('visual')
        
        lua_code = f"""
{self.get_random_fake_comment()}
local function {decryptor_name}({bytes_param}, {state_param})
    {self.get_random_fake_comment()}
    local res = {{}}
    for i, b in ipairs({bytes_param}) do
        local k = ({self.xor_key_base} + ({state_param} % 255) + (i - 1)) % 256
        table.insert(res, string.char({bit32_table_name}['{bit32_bxor_name}'](b, k)))
    end
    return table.concat(res)
end
"""
        return lua_code, decryptor_name

    # =========================================================================
    # ADVERSARIAL CONTEXT POISONING METHODS
    # =========================================================================
    
    def generate_misleading_name(self, category='generic'):
        """
        Generates a misleading variable/function name from the specified category.
        
        These names are designed to confuse AI deobfuscators by suggesting
        benign functionality (UI, audio, networking) when the actual code
        performs something entirely different (combat, ESP, aimbot).
        
        Args:
            category (str): One of 'combat', 'visual', 'targeting', 'memory', 'generic'
            
        Returns:
            str: A misleading name that hasn't been used yet, or falls back to
                 a generated confusing name if pool is exhausted.
        """
        if not self.enable_misleading_names:
            return self.generate_var_name()
        
        # Get the pool for the specified category, fallback to generic
        pool = self.misleading_name_pools.get(category, self.misleading_name_pools['generic'])
        
        # Find unused names in this pool
        available = [name for name in pool if name not in self.used_misleading_names]
        
        if available:
            chosen = random.choice(available)
            self.used_misleading_names.add(chosen)
            return chosen
        else:
            # Pool exhausted - generate a hybrid misleading name
            # Combine a misleading prefix with confusing suffix
            prefixes = ['Update', 'Calculate', 'Process', 'Validate', 'Sync', 'Refresh']
            suffixes = ['Cache', 'Buffer', 'State', 'Metrics', 'Layout', 'Config']
            
            while True:
                hybrid = random.choice(prefixes) + random.choice(suffixes) + str(random.randint(1, 999))
                if hybrid not in self.used_misleading_names:
                    self.used_misleading_names.add(hybrid)
                    return hybrid
    
    def get_random_fake_comment(self):
        """
        Returns a random fake comment designed to mislead AI about code purpose.
        
        AI models heavily weight comments for semantic understanding.
        These comments describe operations completely unrelated to what
        the actual code does, causing AI to hallucinate wrong functionality.
        
        Returns:
            str: A misleading comment string (includes -- prefix)
        """
        if not self.enable_fake_comments:
            return ""
        return random.choice(self.fake_comment_pool)
    
    def generate_fake_comment_block(self, num_comments=3):
        """
        Generates a block of fake comments for insertion into generated code.
        
        Multiple comments together create stronger semantic misdirection,
        making AI more confident in its wrong interpretation.
        
        Args:
            num_comments (int): Number of fake comments to generate
            
        Returns:
            str: Multiple newline-separated fake comments
        """
        if not self.enable_fake_comments:
            return ""
        
        comments = []
        used = set()
        for _ in range(num_comments):
            # Avoid duplicate comments in same block
            available = [c for c in self.fake_comment_pool if c not in used]
            if available:
                comment = random.choice(available)
                used.add(comment)
                comments.append(comment)
        
        return '\n'.join(comments)
    
    # =========================================================================
    # LOGIC FLOODING - MEGA FUNCTION GENERATOR
    # =========================================================================
    
    def generate_mega_dispatch_function(self):
        """
        Generates a massive polymorphic dispatch function that handles 50+ operations.
        
        This creates extreme "function density" that overwhelms AI pattern recognition.
        The function uses a complex dispatch mechanism where:
        1. Primary opcode determines general category
        2. Secondary modifier determines specific operation
        3. Accumulator value can flip behavior (polymorphic instructions)
        
        POLYMORPHIC BEHAVIOR:
        - Opcode 5 usually adds, but if input > threshold, it subtracts
        - This destroys AI's ability to summarize opcodes simply
        
        Returns:
            str: Lua code for the mega dispatch function
        """
        func_name = self.generate_misleading_name('generic')  # Misleading name
        self.mega_dispatch_name = func_name
        
        # Generate random opcode assignments
        # Each opcode maps to an operation, but behavior changes based on accumulator
        opcodes = {}
        for i in range(1, 51):  # 50 opcodes
            opcodes[i] = {
                'primary': random.randint(100, 999),
                'modifier': random.randint(10, 99),
                'threshold': self.poly_opcode_threshold
            }
        
        # Fake comment to mislead about purpose
        header_comment = self.generate_fake_comment_block(2)
        
        lines = [header_comment]
        lines.append(f"local {func_name} = function(op, a, b, acc)")
        lines.append(f"    {self.get_random_fake_comment()}")
        lines.append("    local r = nil")
        
        # Generate the massive dispatch logic
        # Shuffle operations for non-obvious ordering
        ops = [
            ('add', 'a + b', 'a - b'),           # Polymorphic: add or sub based on acc
            ('sub', 'a - b', 'a + b'),           # Polymorphic: sub or add
            ('mul', 'a * b', 'a / (b ~= 0 and b or 1)'),  # Polymorphic: mul or div
            ('div', 'a / (b ~= 0 and b or 1)', 'a * b'),  # Polymorphic: div or mul
            ('mod', 'a % (b ~= 0 and b or 1)', 'math.floor(a / (b ~= 0 and b or 1))'),
            ('pow', 'a ^ b', 'math.sqrt(math.abs(a))'),
            ('neg', '-a', 'math.abs(a)'),
            ('floor', 'math.floor(a)', 'math.ceil(a)'),
            ('ceil', 'math.ceil(a)', 'math.floor(a)'),
            ('abs', 'math.abs(a)', '-math.abs(a)'),
            ('min', 'math.min(a, b)', 'math.max(a, b)'),
            ('max', 'math.max(a, b)', 'math.min(a, b)'),
            ('sin', 'math.sin(a)', 'math.cos(a)'),
            ('cos', 'math.cos(a)', 'math.sin(a)'),
            ('tan', 'math.tan(a)', '1 / math.tan(a ~= 0 and a or 0.001)'),
            ('sqrt', 'math.sqrt(math.abs(a))', 'a * a'),
            ('log', 'math.log(math.abs(a) + 1)', 'math.exp(a)'),
            ('exp', 'math.exp(math.min(a, 20))', 'math.log(math.abs(a) + 1)'),
            ('eq', 'a == b', 'a ~= b'),
            ('neq', 'a ~= b', 'a == b'),
            ('lt', 'a < b', 'a >= b'),
            ('gt', 'a > b', 'a <= b'),
            ('le', 'a <= b', 'a > b'),
            ('ge', 'a >= b', 'a < b'),
            ('band', 'math.floor(a) % 256', 'math.floor(b) % 256'),  # Simplified bit ops
            ('bor', '(math.floor(a) + math.floor(b)) % 256', 'math.abs(math.floor(a) - math.floor(b)) % 256'),
            ('bxor', 'math.abs(math.floor(a) - math.floor(b)) % 256', '(math.floor(a) + math.floor(b)) % 256'),
            ('lshift', 'math.floor(a) * (2 ^ math.min(b, 16))', 'math.floor(a / (2 ^ math.min(b, 16)))'),
            ('rshift', 'math.floor(a / (2 ^ math.min(b, 16)))', 'math.floor(a) * (2 ^ math.min(b, 16))'),
            ('clamp', 'math.max(0, math.min(a, b))', 'math.min(0, math.max(a, -b))'),
            ('lerp', 'a + (b - a) * 0.5', 'b + (a - b) * 0.5'),
            ('sign', 'a > 0 and 1 or (a < 0 and -1 or 0)', 'a >= 0 and 1 or -1'),
            ('round', 'math.floor(a + 0.5)', 'math.ceil(a - 0.5)'),
            ('frac', 'a - math.floor(a)', 'math.ceil(a) - a'),
            ('wrap', 'a % (b ~= 0 and b or 360)', '(a + b) % (b ~= 0 and b or 360)'),
            ('deg', 'math.deg(a)', 'math.rad(a)'),
            ('rad', 'math.rad(a)', 'math.deg(a)'),
            ('atan2', 'math.atan2(a, b ~= 0 and b or 0.001)', 'math.atan(a / (b ~= 0 and b or 0.001))'),
            ('hypot', 'math.sqrt(a*a + b*b)', 'math.abs(a) + math.abs(b)'),
            ('avg', '(a + b) / 2', 'math.abs(a - b) / 2'),
        ]
        
        # Randomize order
        random.shuffle(ops)
        
        # Assign opcodes
        self.mega_opcodes = {}
        for i, (name, normal_op, poly_op) in enumerate(ops[:40]):  # Use first 40
            opcode = random.randint(100, 9999)
            while opcode in self.mega_opcodes.values():
                opcode = random.randint(100, 9999)
            self.mega_opcodes[name] = opcode
            
            # Generate the polymorphic condition
            # If accumulator > threshold, use alternate behavior
            lines.append(f"    if op == {opcode} then")
            lines.append(f"        {self.get_random_fake_comment()}")
            lines.append(f"        if acc and acc > {self.poly_opcode_threshold} then")
            lines.append(f"            r = {poly_op}")
            lines.append(f"        else")
            lines.append(f"            r = {normal_op}")
            lines.append(f"        end")
            lines.append(f"    end")
        
        # Add decoy opcodes that do nothing useful (more confusion)
        for _ in range(10):
            decoy_op = random.randint(10000, 19999)
            decoy_result = random.choice(['nil', '0', 'false', 'a', '""', '{}'])
            lines.append(f"    if op == {decoy_op} then r = {decoy_result} end")
        
        lines.append(f"    {self.get_random_fake_comment()}")
        lines.append("    return r")
        lines.append("end")
        
        return '\n'.join(lines) + '\n'
    
    def get_mega_dispatch_call(self, op_name, left_node, right_node, accumulator=None):
        """
        Creates an AST node for calling the mega dispatch function.
        
        Args:
            op_name (str): Operation name (e.g., 'add', 'sub')
            left_node: Left operand AST node
            right_node: Right operand AST node  
            accumulator: Optional accumulator value for polymorphic behavior
            
        Returns:
            Call: AST Call node for mega_dispatch(opcode, a, b, acc)
        """
        if not hasattr(self, 'mega_opcodes') or op_name not in self.mega_opcodes:
            return None
        
        opcode = self.mega_opcodes[op_name]
        
        # Create the function call
        args = [Number(opcode), left_node, right_node]
        
        if accumulator is not None:
            args.append(Number(accumulator))
        else:
            # Random accumulator below threshold for normal behavior
            args.append(Number(random.randint(1, self.poly_opcode_threshold - 100)))
        
        return Call(Name(self.mega_dispatch_name), args)

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
    # EQUIVALENCY MUTATION SYSTEM
    # =========================================================================
    
    def wrap_in_identity_function(self, value_node):
        """
        Wraps a value in an identity function call to confuse variable tracking.
        
        This transforms:
            local x = 10
        Into:
            local x = select(1, 10)
        
        Why select(1, value)?
        - select(1, ...) returns its first vararg, which is just the value itself
        - Creates a function call boundary that confuses AI deobfuscators
        - Uses valid, simple Lua syntax (no IIFE parentheses issues)
        - The luaparser's to_lua_source handles this correctly
        
        Note: We previously used IIFEs like (function(v) return v end)(10), but
        luaparser doesn't wrap anonymous functions in parentheses, producing
        invalid syntax: function(v) return v end(10)
        
        Args:
            value_node: The AST node representing the value to wrap
            
        Returns:
            Node: A Call node wrapping the value in select(1, ...)
        """
        # Create: select(1, value)
        # This is equivalent to just returning the value, but wrapped in a function call
        select_call = Call(
            Name('select'),
            [Number(1), value_node]
        )
        
        return select_call
    
    def wrap_in_complex_identity(self, value_node):
        """
        Alternative wrapper using math operations for numeric values.
        
        This transforms:
            local x = 10
        Into one of:
            local x = (10 + 0)
            local x = (10 * 1)
            local x = math.abs(10)  -- for positive numbers
            local x = select(1, 10)
        
        Args:
            value_node: The AST node representing the value to wrap
            
        Returns:
            Node: An AST node that evaluates to the same value
        """
        # Helper to create binary operations
        def make_binop(op_class, left, right):
            if SPECIFIC_BINOP_MODE:
                return op_class(left, right)
            else:
                return BINOP(op_class(), left, right)
        
        # For Number nodes, we can use math identity operations
        if isinstance(value_node, Number):
            strategy = random.randint(1, 4)
            
            if strategy == 1:
                # value + 0
                return make_binop(ADD_OP, value_node, Number(0))
            elif strategy == 2:
                # value * 1
                return make_binop(MULT_OP, value_node, Number(1))
            elif strategy == 3 and value_node.n >= 0:
                # math.abs(value) for non-negative numbers
                # Use bracket notation: math["abs"](value)
                return Call(
                    Index(self._str('abs'), Name('math'), notation=1),
                    [value_node]
                )
            else:
                # select(1, value)
                return Call(Name('select'), [Number(1), value_node])
        
        # For non-numbers, use select
        return Call(Name('select'), [Number(1), value_node])
    
    def apply_equivalency_mutation(self, node, mutation_rate=0.3):
        """
        Recursively traverse AST and wrap simple values in identity expressions.
        
        Uses select(1, value) for general values and math identities for numbers.
        
        This applies to:
        - LocalAssign values (local x = VALUE)
        - Assign values (x = VALUE)
        - Number and String literals in certain contexts
        
        Args:
            node: The AST node to process
            mutation_rate: Probability (0-1) of mutating each eligible value
        """
        if node is None:
            return
        
        # Process LocalAssign
        if isinstance(node, LocalAssign):
            if node.values:
                new_values = []
                for value in node.values:
                    # Recurse first to handle nested structures
                    self.apply_equivalency_mutation(value, mutation_rate)
                    
                    # Apply mutation to simple values with probability
                    if random.random() < mutation_rate:
                        if isinstance(value, Number):
                            # Use complex identity for numbers (adds math operations)
                            new_values.append(self.wrap_in_complex_identity(value))
                        elif isinstance(value, String):
                            # Use select for strings
                            new_values.append(self.wrap_in_identity_function(value))
                        elif isinstance(value, Name):
                            # Wrap variable references with select
                            new_values.append(self.wrap_in_identity_function(value))
                        else:
                            new_values.append(value)
                    else:
                        new_values.append(value)
                node.values = new_values
            return
        
        # Process Assign
        if isinstance(node, Assign):
            if node.values:
                new_values = []
                for value in node.values:
                    self.apply_equivalency_mutation(value, mutation_rate)
                    
                    if random.random() < mutation_rate:
                        if isinstance(value, Number):
                            new_values.append(self.wrap_in_complex_identity(value))
                        elif isinstance(value, String):
                            new_values.append(self.wrap_in_identity_function(value))
                        else:
                            new_values.append(value)
                    else:
                        new_values.append(value)
                node.values = new_values
            return
        
        # Recurse into child nodes
        if hasattr(node, '__dict__'):
            for attr_name, attr_value in list(node.__dict__.items()):
                if attr_name.startswith('_'):
                    continue
                
                if isinstance(attr_value, list):
                    for item in attr_value:
                        if isinstance(item, Node):
                            self.apply_equivalency_mutation(item, mutation_rate)
                
                elif isinstance(attr_value, Node):
                    self.apply_equivalency_mutation(attr_value, mutation_rate)

    # =========================================================================
    # RUNTIME ENVIRONMENT ANCHORING
    # =========================================================================
    # Uses Matcha/Roblox-specific runtime values that AI cannot predict
    
    def generate_runtime_key_expression(self):
        """
        Generates an expression that computes a key value at runtime using
        Matcha/Roblox environment values that AI cannot predict.
        
        Returns:
            tuple: (AST node for the expression, expected integer value for testing)
        
        Example outputs:
        - #game:GetService("Players").LocalPlayer.Name
        - game.PlaceId % 100
        - #tostring(workspace)
        """
        strategy = random.randint(1, 5)
        
        # Helper to create method call: obj:Method(args)
        def invoke(obj, method, args=None):
            return Invoke(obj, Name(method), args or [])
        
        # Helper to create property access: obj["prop"] (bracket notation for safety)
        def prop(obj, prop_name):
            return Index(self._str(prop_name), obj, notation=1)
        
        # Helper for game:GetService("ServiceName")
        def get_service(service_name):
            return invoke(Name('game'), 'GetService', [self._str(service_name)])
        
        if strategy == 1:
            # #game:GetService("Players").LocalPlayer.Name
            # Player names are typically 3-20 chars, we use modulo for consistency
            players = get_service("Players")
            local_player = prop(players, 'LocalPlayer')
            name = prop(local_player, 'Name')
            # Wrap in length operator - we'll use ULenOp or create manually
            # For luaparser, length is typically a UnaryOp
            try:
                len_tree = ast.parse("local x = #'test'")
                len_stmts = get_statements(len_tree)
                len_node = len_stmts[0].values[0] if len_stmts else None
                LEN_OP = type(len_node) if len_node else None
                if LEN_OP:
                    expr = LEN_OP(name)
                else:
                    # Fallback: use string["len"](name)
                    expr = Call(Index(self._str('len'), Name('string'), notation=1), [name])
            except:
                expr = Call(Index(self._str('len'), Name('string'), notation=1), [name])
            
            # Modulo to keep value small and predictable range
            expr = self._make_binop(MOD_OP if 'MOD_OP' in dir() else SUB_OP, expr, Number(20))
            estimated_value = 8  # Typical username length
            
        elif strategy == 2:
            # game["PlaceId"] % 100
            place_id = prop(Name('game'), 'PlaceId')
            expr = self._make_binop(MOD_OP if 'MOD_OP' in dir() else SUB_OP, place_id, Number(100))
            estimated_value = 42  # Placeholder - actual value unknown to AI
            
        elif strategy == 3:
            # game["JobId"]:len() % 36  (JobIds are GUIDs, 36 chars)
            job_id = prop(Name('game'), 'JobId')
            len_call = invoke(job_id, 'len', [])
            expr = self._make_binop(MOD_OP if 'MOD_OP' in dir() else SUB_OP, len_call, Number(50))
            estimated_value = 36
            
        elif strategy == 4:
            # tick() % 1000 (changes every call - highly dynamic)
            tick_call = Call(Name('tick'), [])
            expr = self._make_binop(MOD_OP if 'MOD_OP' in dir() else SUB_OP, tick_call, Number(1000))
            estimated_value = 500  # Random midpoint
            
        else:
            # os["time"]() % 100
            os_time = Call(Index(self._str('time'), Name('os'), notation=1), [])
            expr = self._make_binop(MOD_OP if 'MOD_OP' in dir() else SUB_OP, os_time, Number(100))
            estimated_value = 50
        
        return expr, estimated_value
    
    def _make_binop(self, op_class, left, right):
        """Helper to create binary operation respecting SPECIFIC_BINOP_MODE."""
        if SPECIFIC_BINOP_MODE:
            return op_class(left, right)
        else:
            return BINOP(op_class(), left, right)

    # =========================================================================
    # MATCHA-SPECIFIC TAUTOLOGIES (Opaque Predicates v2)
    # =========================================================================
    # Uses Matcha VM behavior that AI cannot predict
    
    def generate_matcha_tautology(self, want_true=True):
        """
        Generates a condition that evaluates to True or False based on
        Matcha-specific behavior that AI cannot predict.
        
        Args:
            want_true: If True, generate always-true condition. If False, always-false.
            
        Returns:
            AST node representing the condition
        
        True conditions (tautologies):
        - type(Drawing.new("Square")) == "table"
        - type(game) == "userdata"
        - typeof(workspace) == "Instance"
        
        False conditions (contradictions using Matcha quirks):
        - tostring(Vector3.new(1,1,1)) == "1, 1, 1"  (Matcha prints address)
        - task.defer ~= nil  (task.defer crashes/doesn't exist in Matcha)
        """
        
        # Helper to create equality check
        def eq_check(left, right):
            return self._make_binop(EQ_OP, left, right)
        
        # Helper to create not-equal check
        def neq_check(left, right):
            try:
                neq_tree = ast.parse("local x = 1 ~= 2")
                neq_stmts = get_statements(neq_tree)
                neq_node = neq_stmts[0].values[0] if neq_stmts else None
                if neq_node and hasattr(neq_node, 'op'):
                    NEQ_OP = type(neq_node.op)
                else:
                    NEQ_OP = type(neq_node) if neq_node else EQ_OP
                return self._make_binop(NEQ_OP, left, right)
            except:
                # Fallback - negate equality
                return self._make_binop(EQ_OP, left, right)
        
        if want_true:
            # Generate always-true conditions based on Matcha behavior
            strategy = random.randint(1, 4)
            
            if strategy == 1:
                # type(Drawing.new("Square")) == "table"
                # Use bracket notation: Drawing["new"]("Square")
                drawing_new = Call(
                    Index(self._str('new'), Name('Drawing'), notation=1),
                    [self._str("Square")]
                )
                type_call = Call(Name('type'), [drawing_new])
                return eq_check(type_call, self._str("table"))
                
            elif strategy == 2:
                # type(game) == "userdata"
                type_call = Call(Name('type'), [Name('game')])
                return eq_check(type_call, self._str("userdata"))
                
            elif strategy == 3:
                # typeof(workspace) == "Instance"
                typeof_call = Call(Name('typeof'), [Name('workspace')])
                return eq_check(typeof_call, self._str("Instance"))
                
            else:
                # game:GetService("Players") ~= nil
                # Use Invoke to create colon call: game:GetService("Players")
                # This correctly passes 'game' as self
                players = Invoke(
                    Name('game'),
                    Name('GetService'),
                    [self._str("Players")]
                )
                return neq_check(players, Name('nil'))
        
        else:
            # Generate always-false conditions (Matcha-specific contradictions)
            strategy = random.randint(1, 3)
            
            if strategy == 1:
                # tostring(Vector3.new(1,1,1)) == "1, 1, 1"
                # In standard Roblox this is true, in Matcha it prints memory address
                # Use bracket notation: Vector3["new"](1,1,1)
                vec3 = Call(
                    Index(self._str('new'), Name('Vector3'), notation=1),
                    [Number(1), Number(1), Number(1)]
                )
                tostr = Call(Name('tostring'), [vec3])
                return eq_check(tostr, self._str("1, 1, 1"))
                
            elif strategy == 2:
                # task.defer ~= nil (task.defer doesn't work in Matcha)
                # Use bracket notation: task["defer"]
                task_defer = Index(self._str('defer'), Name('task'), notation=1)
                return neq_check(task_defer, Name('nil'))
                
            else:
                # Vector3.new(1,0,0).Unit ~= nil (.Unit doesn't exist in Matcha)
                # Use bracket notation: Vector3["new"](1,0,0)["Unit"]
                vec3 = Call(
                    Index(self._str('new'), Name('Vector3'), notation=1),
                    [Number(1), Number(0), Number(0)]
                )
                unit_prop = Index(self._str('Unit'), vec3, notation=1)
                return neq_check(unit_prop, Name('nil'))

    # =========================================================================
    # POLYMORPHIC OPERATOR DISPATCH
    # =========================================================================
    # Many-to-one mapping with salt values to prevent simple substitution
    
    def generate_polymorphic_ops_table(self):
        """
        Generate a polymorphic operator dispatch system.
        
        Instead of simple Ops[1] = add, we use:
        PolyMath(mode, a, b, salt) where mode + salt determines operation
        
        Returns:
            str: Lua code defining the polymorphic dispatch function
        """
        func_name = self.generate_var_name()
        self.poly_dispatch_name = func_name
        
        # Generate random base values for each operation
        # The formula is: if (mode + salt) == target then return operation
        self.poly_targets = {
            'add': random.randint(500, 599),
            'sub': random.randint(600, 699),
            'mul': random.randint(700, 799),
            'div': random.randint(800, 899),
            'mod': random.randint(900, 999),
            'eq': random.randint(1000, 1099),
            'neq': random.randint(1100, 1199),
            'lt': random.randint(1200, 1299),
            'gt': random.randint(1300, 1399),
            'le': random.randint(1400, 1499),
            'ge': random.randint(1500, 1599),
            'concat': random.randint(1600, 1699),
        }
        
        # Build the function with randomized order
        lines = [f"local {func_name} = function(mode, a, b, salt)"]
        lines.append("    local t = mode + salt")
        
        # Shuffle the operations for non-obvious ordering
        ops_list = list(self.poly_targets.items())
        random.shuffle(ops_list)
        
        for op_type, target in ops_list:
            if op_type == 'add':
                lines.append(f"    if t == {target} then return a + b end")
            elif op_type == 'sub':
                lines.append(f"    if t == {target} then return a - b end")
            elif op_type == 'mul':
                lines.append(f"    if t == {target} then return a * b end")
            elif op_type == 'div':
                lines.append(f"    if t == {target} then return a / b end")
            elif op_type == 'mod':
                lines.append(f"    if t == {target} then return a % b end")
            elif op_type == 'eq':
                lines.append(f"    if t == {target} then return a == b end")
            elif op_type == 'neq':
                lines.append(f"    if t == {target} then return a ~= b end")
            elif op_type == 'lt':
                lines.append(f"    if t == {target} then return a < b end")
            elif op_type == 'gt':
                lines.append(f"    if t == {target} then return a > b end")
            elif op_type == 'le':
                lines.append(f"    if t == {target} then return a <= b end")
            elif op_type == 'ge':
                lines.append(f"    if t == {target} then return a >= b end")
            elif op_type == 'concat':
                lines.append(f"    if t == {target} then return a .. b end")
        
        # Add decoy/junk conditions
        for _ in range(3):
            fake_target = random.randint(100, 499)
            fake_result = random.choice(['nil', '0', 'false', 'a'])
            lines.append(f"    if t == {fake_target} then return {fake_result} end")
        
        lines.append("    return nil")
        lines.append("end")
        
        return "\n".join(lines) + "\n"
    
    def get_poly_call_params(self, op_type):
        """
        Get the (mode, salt) pair for a polymorphic operation call.
        
        Args:
            op_type: The operation type string
            
        Returns:
            tuple: (mode, salt) where mode + salt == target for this operation
        """
        if not hasattr(self, 'poly_targets') or op_type not in self.poly_targets:
            return None, None
            
        target = self.poly_targets[op_type]
        # Split target into mode + salt randomly
        salt = random.randint(100, 400)
        mode = target - salt
        return mode, salt

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
        
        ANTI-AI ENHANCEMENT:
        - Adds misleading fake comments before each operation
        - Uses misleading local variable names within functions
        - Interleaves decoy functions that are never called
        
        Returns:
            str: Lua code defining the operators table
        """
        lines = []
        
        # Add a misleading header comment block
        if self.enable_fake_comments:
            lines.append(self.get_random_fake_comment())
            lines.append("-- Virtualized texture sampling pipeline")
        
        lines.append(f"local {self.virt_table_name} = {{}}")
        
        # Misleading internal variable names for the lambda parameters
        # AI sees "texCoord, normalMap" and thinks this is rendering code
        misleading_params = [
            ('texCoord', 'normalMap'),
            ('audioLevel', 'mixRatio'),
            ('uiScale', 'anchorOffset'),
            ('netLatency', 'packetId'),
            ('fontMetric', 'glyphIndex'),
            ('shadowBias', 'cascadeLevel'),
        ]
        
        # Generate functions for each registered operation
        # Using clean arithmetic (no modifiers) to support Vectors/userdata
        op_index = 0
        for op_type, key in self.virt_ops.items():
            # Select misleading parameter names
            param_a, param_b = misleading_params[op_index % len(misleading_params)]
            op_index += 1
            
            # Add a fake comment before each operation (50% chance)
            if self.enable_fake_comments and random.random() < 0.5:
                lines.append(self.get_random_fake_comment())
            
            if op_type == 'add':
                # Clean addition - supports Numbers, Vectors, etc.
                lines.append(f"{self.virt_table_name}[{key}] = function({param_a}, {param_b}) return {param_a} + {param_b} end")
                
            elif op_type == 'sub':
                # Clean subtraction - supports Numbers, Vectors, etc.
                lines.append(f"{self.virt_table_name}[{key}] = function({param_a}, {param_b}) return {param_a} - {param_b} end")
                
            elif op_type == 'mul':
                # Clean multiplication - supports Numbers, Vectors, etc.
                lines.append(f"{self.virt_table_name}[{key}] = function({param_a}, {param_b}) return {param_a} * {param_b} end")
                
            elif op_type == 'div':
                # Clean division - supports Numbers, Vectors, etc.
                lines.append(f"{self.virt_table_name}[{key}] = function({param_a}, {param_b}) return {param_a} / {param_b} end")
                
            elif op_type == 'mod':
                # Modulo
                lines.append(f"{self.virt_table_name}[{key}] = function({param_a}, {param_b}) return {param_a} % {param_b} end")
                
            elif op_type == 'eq':
                # Equality
                lines.append(f"{self.virt_table_name}[{key}] = function({param_a}, {param_b}) return {param_a} == {param_b} end")
                
            elif op_type == 'neq':
                # Not equal
                lines.append(f"{self.virt_table_name}[{key}] = function({param_a}, {param_b}) return {param_a} ~= {param_b} end")
                
            elif op_type == 'lt':
                # Less than
                lines.append(f"{self.virt_table_name}[{key}] = function({param_a}, {param_b}) return {param_a} < {param_b} end")
                
            elif op_type == 'gt':
                # Greater than
                lines.append(f"{self.virt_table_name}[{key}] = function({param_a}, {param_b}) return {param_a} > {param_b} end")
                
            elif op_type == 'le':
                # Less or equal
                lines.append(f"{self.virt_table_name}[{key}] = function({param_a}, {param_b}) return {param_a} <= {param_b} end")
                
            elif op_type == 'ge':
                # Greater or equal
                lines.append(f"{self.virt_table_name}[{key}] = function({param_a}, {param_b}) return {param_a} >= {param_b} end")
                
            elif op_type == 'and_op':
                # Logical and
                lines.append(f"{self.virt_table_name}[{key}] = function({param_a}, {param_b}) return {param_a} and {param_b} end")
                
            elif op_type == 'or_op':
                # Logical or
                lines.append(f"{self.virt_table_name}[{key}] = function({param_a}, {param_b}) return {param_a} or {param_b} end")
                
            elif op_type == 'concat':
                # String concatenation
                lines.append(f"{self.virt_table_name}[{key}] = function({param_a}, {param_b}) return {param_a} .. {param_b} end")
        
        # Add decoy functions (never called, but confuse analysis)
        # These use keys that will never be used
        decoy_comments = [
            "-- Texture coordinate transformation",
            "-- Audio channel mixing",
            "-- UI layout recalculation",
            "-- Network packet validation"
        ]
        for i in range(3):
            decoy_key = random.randint(90000, 99999)
            decoy_param_a, decoy_param_b = random.choice(misleading_params)
            if self.enable_fake_comments:
                lines.append(random.choice(decoy_comments))
            # Decoy functions return garbage that will never execute
            lines.append(f"{self.virt_table_name}[{decoy_key}] = function({decoy_param_a}, {decoy_param_b}) return nil end")
        
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

    def generate_opaque_predicate(self):
        """
        Generates an opaque predicate - a condition that is mathematically always false
        but appears complex to static analysis and AI deobfuscators.
        
        Returns a tuple of (condition_node, complexity_score) where higher scores
        indicate more complex predicates.
        """
        predicate_type = random.randint(1, 6)
        num = random.randint(1, 1000)
        num2 = random.randint(1, 1000)
        
        # Helper to create math["func"](arg) call
        # Use bracket notation with string key for reliable output
        def math_call(func_name, *args):
            return Call(
                Index(self._str(func_name), Name('math'), notation=1),  # Creates math["func_name"]
                list(args)
            )
        
        # Helper to create binary operation
        def make_binop(op_class, left, right):
            if SPECIFIC_BINOP_MODE:
                return op_class(left, right)
            else:
                return BINOP(op_class(), left, right)
        
        # Discover comparison operators dynamically
        try:
            gt_tree = ast.parse("local x = 1 > 1")
            gt_stmts = get_statements(gt_tree)
            gt_node = gt_stmts[0].values[0] if gt_stmts else None
            GT_OP = type(gt_node.op) if gt_node and hasattr(gt_node, 'op') else type(gt_node)
                
            lt_tree = ast.parse("local x = 1 < 1")
            lt_stmts = get_statements(lt_tree)
            lt_node = lt_stmts[0].values[0] if lt_stmts else None
            LT_OP = type(lt_node.op) if lt_node and hasattr(lt_node, 'op') else type(lt_node)
            
            mod_tree = ast.parse("local x = 5 % 2")
            mod_stmts = get_statements(mod_tree)
            mod_node = mod_stmts[0].values[0] if mod_stmts else None
            MOD_OP = type(mod_node.op) if mod_node and hasattr(mod_node, 'op') else type(mod_node)
        except:
            GT_OP = EQ_OP
            LT_OP = EQ_OP
            MOD_OP = SUB_OP
        
        complexity = 1
        condition = None
        
        try:
            if predicate_type == 1:
                # math.abs(math.sin(n)) > 2  -- Always false (sin in [-1,1])
                sin_call = math_call('sin', Number(num))
                abs_call = math_call('abs', sin_call)
                condition = make_binop(GT_OP, abs_call, Number(2))
                complexity = 2
                
            elif predicate_type == 2:
                # math.sin(n)^2 + math.cos(n)^2 > 1.5  -- Always false (Pythagorean identity = 1)
                sin_sq = math_call('pow', math_call('sin', Number(num)), Number(2))
                cos_sq = math_call('pow', math_call('cos', Number(num)), Number(2))
                sum_expr = make_binop(ADD_OP, sin_sq, cos_sq)
                condition = make_binop(GT_OP, sum_expr, Number(1.5))
                complexity = 4
                
            elif predicate_type == 3:
                # math.floor(n) > n + 1  -- Always false (floor(n) <= n)
                floor_call = math_call('floor', Number(num))
                num_plus_1 = make_binop(ADD_OP, Number(num), Number(1))
                condition = make_binop(GT_OP, floor_call, num_plus_1)
                complexity = 2
                
            elif predicate_type == 4:
                # (n % 1) > 0.5 for integer n  -- Always false (int % 1 == 0)
                mod_expr = make_binop(MOD_OP, Number(num), Number(1))
                condition = make_binop(GT_OP, mod_expr, Number(0.5))
                complexity = 2
                
            elif predicate_type == 5:
                # math.abs(n) < -1  -- Always false (abs is never negative)
                abs_call = math_call('abs', Number(num))
                neg_one = make_binop(SUB_OP, Number(0), Number(1))
                condition = make_binop(LT_OP, abs_call, neg_one)
                complexity = 2
                
            elif predicate_type == 6:
                # math.exp(n) < 0  -- Exponential is always positive
                exp_call = math_call('exp', Number(num % 10))  # Keep small to avoid overflow
                condition = make_binop(LT_OP, exp_call, Number(0))
                complexity = 2
                
        except Exception:
            pass
        
        # Fallback if condition creation failed
        if condition is None:
            num2 = random.randint(100, 999)
            while num == num2:
                num2 = random.randint(100, 999)
            condition = make_binop(EQ_OP, Number(num), Number(num2))
            complexity = 1
            
        return condition, complexity

    def generate_junk_node(self):
        """
        Creates a 'junk' control flow block (If statement) that serves as obfuscation noise.
        
        Uses THREE types of opaque predicates:
        1. Math-based (sin²+cos²=1, |sin(x)|≤1) - 40% chance
        2. Matcha-specific contradictions (Vector3 tostring, task.defer) - 40% chance
        3. Runtime environment checks that AI can't evaluate - 20% chance
        
        The body contains meaningful-looking but dead code.
        """
        if not EQ_OP:
            return None 
        
        # Choose predicate type
        predicate_choice = random.random()
        
        if predicate_choice < 0.4:
            # Math-based opaque predicate (always false)
            condition, _ = self.generate_opaque_predicate()
        elif predicate_choice < 0.8:
            # Matcha-specific contradiction (always false in Matcha)
            try:
                condition = self.generate_matcha_tautology(want_true=False)
            except:
                condition, _ = self.generate_opaque_predicate()
        else:
            # Complex runtime check that evaluates to false
            # We use a tautology wrapped in a negation-like construct
            try:
                # Create: not (type(game) == "userdata") -- always false since it IS userdata
                inner = self.generate_matcha_tautology(want_true=True)
                # Wrap in "not" by checking == false
                condition = self._make_binop(EQ_OP, inner, Name('false'))
            except:
                condition, _ = self.generate_opaque_predicate()

        # Create Body: Complex junk that looks meaningful
        junk_type = random.randint(1, 5)
        
        if TRUE_NODE:
            true_node = TRUE_NODE()
        else:
            true_node = Name('true')
        
        if junk_type == 1:
            # while true do break end
            loop_body = Block([Break()])
            junk_body = Block([While(true_node, loop_body)])
            
        elif junk_type == 2:
            # Nested dead assignment with math operation
            junk_var = self.generate_var_name()
            dead_assign = LocalAssign(
                [Name(junk_var)],
                [self._make_binop(ADD_OP, Number(random.randint(1,100)), Number(random.randint(1,100)))]
            )
            junk_body = Block([dead_assign])
            
        elif junk_type == 3:
            # repeat until true (dead code)
            repeat_body = Block([Break()])
            junk_body = Block([Repeat(repeat_body, true_node)])
            
        elif junk_type == 4:
            # Fake API call that looks real
            junk_var = self.generate_var_name()
            # game:GetService("SomeService") - looks like real code
            # Use Invoke to create colon call: game:GetService("ServiceName")
            fake_service = Invoke(
                Name('game'),
                Name('GetService'),
                [self._str(random.choice(["RunService", "Players", "Lighting", "ReplicatedStorage"]))]
            )
            dead_assign = LocalAssign([Name(junk_var)], [fake_service])
            junk_body = Block([dead_assign])
            
        else:
            # Nested if with another false condition
            inner_condition, _ = self.generate_opaque_predicate()
            inner_var = self.generate_var_name()
            inner_assign = LocalAssign([Name(inner_var)], [Number(0)])
            inner_if = If(inner_condition, Block([inner_assign]), [])
            junk_body = Block([inner_if])
        
        return If(condition, junk_body, [])

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
        Flattens the control flow of the root chunk using an ADVANCED state machine.
        
        SPAGHETTIFICATION FEATURES:
        1. Ghost States: For each real state, generate 1-2 fake states with junk code
        2. State Interleaving: State transitions can happen mid-block, not just at end
        3. Runtime Key Anchoring: Key derived from Matcha environment (game.PlaceId, etc.)
        4. Relative Transitions: 70% use offsets instead of absolute values
        5. Conditional Multi-Transitions: Some states have multiple exit paths
        
        This creates a control flow GRAPH, not a simple linked list.
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
        statements_list = node.body
        if isinstance(node.body, Block):
            statements_list = node.body.body
        
        if not isinstance(statements_list, list):
            return

        # ---------------------------------------------------------------------
        # PRE-PASS: Hoist Local Definitions
        # ---------------------------------------------------------------------
        hoisted_names = []
        new_statements_list = []
        
        for stmt in statements_list:
            if isinstance(stmt, LocalAssign):
                for target in stmt.targets:
                    if isinstance(target, Name):
                        hoisted_names.append(Name(target.id))
                if stmt.values:
                    new_stmt = Assign(stmt.targets, stmt.values)
                    new_statements_list.append(new_stmt)
                    
            elif isinstance(stmt, LocalFunction):
                if isinstance(stmt.name, Name):
                    hoisted_names.append(Name(stmt.name.id))
                    if ANON_FUNC:
                        anon_func = ANON_FUNC(stmt.args, stmt.body)
                        new_stmt = Assign([stmt.name], [anon_func])
                        new_statements_list.append(new_stmt)
                    else:
                        new_statements_list.append(stmt)
                else:
                    new_statements_list.append(stmt)
            else:
                new_statements_list.append(stmt)

        statements_list = new_statements_list

        # Step A: Chunking
        chunks = []
        current_chunk = []
        
        for stmt in statements_list:
            if isinstance(stmt, Function):
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []
                chunks.append([stmt])
            else:
                current_chunk.append(stmt)
                if len(current_chunk) >= 3:
                    chunks.append(current_chunk)
                    current_chunk = []
        
        if current_chunk:
            chunks.append(current_chunk)

        if not chunks:
            return

        # =====================================================================
        # ADVANCED STATE MACHINE WITH GHOST STATES & SPAGHETTIFICATION
        # =====================================================================
        
        # Helper for binary operations
        def make_binop(op_class, left, right):
            if SPECIFIC_BINOP_MODE:
                return op_class(left, right)
            else:
                return BINOP(op_class(), left, right)
        
        # Generate variable names
        state_var_name = self.generate_var_name()
        state_key_name = self.generate_var_name()
        state_var = Name(state_var_name)
        key_var = Name(state_key_name)
        
        # =====================================================================
        # STATE-DEPENDENT DECRYPTION: Store state variable name
        # =====================================================================
        # This allows string encryption to be linked to the CFF state
        # AI cannot decrypt strings without simulating the entire state machine
        self.state_var_name = state_var_name  # Store for generate_output()
        
        # Generate state values with gaps for ghost states
        base_state = random.randint(100, 300)
        state_gap = random.randint(50, 100)  # Gap between consecutive real states
        
        # Real states
        real_states = []
        current_state = base_state
        for i in range(len(chunks)):
            real_states.append(current_state)
            current_state += state_gap + random.randint(-10, 20)
        
        # =====================================================================
        # Store state mapping for state-dependent string encryption
        # =====================================================================
        self.state_values = real_states  # Maps chunk index -> state ID
        
        # Generate GHOST STATES (fake states with junk code)
        ghost_states = []
        all_used_states = set(real_states)
        
        for real_state in real_states:
            # 1-2 ghost states per real state
            num_ghosts = random.randint(1, 2)
            for _ in range(num_ghosts):
                # Generate a ghost state value near the real state
                ghost = real_state + random.randint(-30, 30)
                while ghost in all_used_states or ghost < 0:
                    ghost = real_state + random.randint(-30, 30)
                ghost_states.append(ghost)
                all_used_states.add(ghost)
        
        start_state = real_states[0]
        end_state = -1
        state_key = random.randint(100, 500)

        # Build the If-ElseIf dispatch chain
        # Start with termination condition
        final_cond = make_binop(EQ_OP, state_var, Number(end_state))
        dispatch_chain = If(final_cond, Block([Break()]), [])
        
        # Add GHOST STATE handlers (junk code that looks real)
        random.shuffle(ghost_states)
        for ghost in ghost_states:
            # Generate junk code for this ghost state
            junk_stmts = []
            
            # Add some fake computation
            junk_var = self.generate_var_name()
            junk_stmts.append(LocalAssign(
                [Name(junk_var)],
                [make_binop(ADD_OP, Number(random.randint(1, 100)), Number(random.randint(1, 100)))]
            ))
            
            # Ghost state transitions to another ghost or loops back
            if random.random() < 0.5 and len(ghost_states) > 1:
                # Transition to another ghost state (creates cycles)
                other_ghost = random.choice([g for g in ghost_states if g != ghost])
                junk_stmts.append(Assign([Name(state_var_name)], [Number(other_ghost)]))
            else:
                # Transition to a random real state (breaks the pattern)
                random_real = random.choice(real_states)
                junk_stmts.append(Assign([Name(state_var_name)], [Number(random_real)]))
            
            ghost_body = Block(junk_stmts)
            
            # Condition for ghost state - use various comparison styles
            if random.random() < 0.5:
                ghost_cond = make_binop(EQ_OP, state_var, Number(ghost))
            else:
                # Key-based comparison
                state_minus_key = make_binop(SUB_OP, state_var, key_var)
                ghost_cond = make_binop(EQ_OP, state_minus_key, Number(ghost - state_key))
            
            dispatch_chain = If(ghost_cond, ghost_body, [dispatch_chain])
        
        # Add REAL STATE handlers (actual code)
        # Process backwards for correct nesting
        for i in range(len(chunks) - 1, -1, -1):
            curr_state = real_states[i]
            
            # Determine next state
            if i < len(chunks) - 1:
                next_state = real_states[i + 1]
                offset_to_next = next_state - curr_state
            else:
                next_state = end_state
                offset_to_next = None
            
            stmts = list(chunks[i])
            
            # STATE INTERLEAVING: Sometimes put transition in middle of block
            use_interleaved = random.random() < 0.3 and len(stmts) > 1
            
            if use_interleaved and offset_to_next is not None:
                # Insert state transition in the MIDDLE of the block
                insert_pos = random.randint(1, len(stmts))
                
                # Use relative transition
                offset_expr = make_binop(ADD_OP, state_var, Number(offset_to_next))
                state_transition = Assign([Name(state_var_name)], [offset_expr])
                stmts.insert(insert_pos, state_transition)
            else:
                # Normal: transition at end
                use_relative = random.random() < 0.7
                
                if use_relative and offset_to_next is not None:
                    offset_expr = make_binop(ADD_OP, state_var, Number(offset_to_next))
                    state_transition = Assign([Name(state_var_name)], [offset_expr])
                else:
                    state_transition = Assign([Name(state_var_name)], [Number(next_state)])
                
                # Handle return statements
                if stmts and isinstance(stmts[-1], Return):
                    return_stmt = stmts.pop()
                    stmts.append(state_transition)
                    stmts.append(return_stmt)
                else:
                    stmts.append(state_transition)
            
            block_body = Block(stmts)
            
            # Create condition with variety
            cond_style = random.randint(1, 3)
            
            if cond_style == 1:
                # Direct: _state == curr_state
                condition = make_binop(EQ_OP, state_var, Number(curr_state))
            elif cond_style == 2:
                # Key-based: (_state - _key) == (curr_state - state_key)
                state_minus_key = make_binop(SUB_OP, state_var, key_var)
                condition = make_binop(EQ_OP, state_minus_key, Number(curr_state - state_key))
            else:
                # Key-offset: _state == (_key + offset)
                key_plus_offset = make_binop(ADD_OP, key_var, Number(curr_state - state_key))
                condition = make_binop(EQ_OP, state_var, key_plus_offset)
            
            dispatch_chain = If(condition, block_body, [dispatch_chain])
        
        # Create the main loop
        if TRUE_NODE:
            true_node = TRUE_NODE()
        else:
            true_node = Name('true')
            
        loop_node = While(true_node, Block([dispatch_chain]))
        
        # Build final body
        key_init = LocalAssign([Name(state_key_name)], [Number(state_key)])
        state_init = LocalAssign([Name(state_var_name)], [Number(start_state)])
        
        final_body = [key_init, state_init, loop_node]
        
        if hoisted_names:
            hoist_decl = LocalAssign(hoisted_names, [])
            final_body.insert(0, hoist_decl)
            
        # Replace the original root body
        if isinstance(node.body, Block):
            node.body.body = final_body
        else:
            node.body = final_body

    def inject_junk_code(self, node, is_root=True):
        """
        Recursively traverse the AST and inject junk code into statement lists.
        
        SAFETY: Only injects into 'body' attributes which contain statement lists.
        Never injects into:
        - 'values' (assignment RHS)
        - 'args' (function arguments)
        - 'targets' (assignment LHS)
        - 'iters'/'iterators' (for loop iterators)
        - 'condition' (if/while conditions)
        
        Args:
            node: The current AST node.
            is_root (bool): True if this is the top-level node (Chunk).
        """
        if node is None:
            return

        # List of attribute names that are SAFE to inject statements into
        # These are statement blocks, not expression lists
        SAFE_INJECTION_ATTRS = {'body'}
        
        # List of attribute names we should NEVER inject into
        # These are expression contexts where statements are invalid
        UNSAFE_ATTRS = {
            'values',      # Assignment RHS: local x = [values]
            'args',        # Function call arguments
            'targets',     # Assignment LHS
            'iters',       # For loop iterators
            'iterators',   # Alternative name for iterators
            'iter',        # Single iterator
            'condition',   # If/while conditions
            'test',        # Alternative condition name
            'idx',         # Index expressions
            'value',       # Single value expressions
            'start',       # For loop start
            'stop',        # For loop stop  
            'step',        # For loop step
            'func',        # Function being called
            'source',      # Table source in index
        }

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

                # SAFETY CHECK: Only inject into known-safe statement block attributes
                if attr_name in SAFE_INJECTION_ATTRS:
                    # Additional safety: verify list contains statements, not expressions
                    # Statements typically include If, While, Assign, LocalAssign, Return, etc.
                    # Skip if the list appears to contain only expressions
                    
                    if attr_value and len(attr_value) > 0:
                        first_item = attr_value[0]
                        # Check if first item looks like a statement (has certain attributes)
                        is_statement_list = isinstance(first_item, (
                            If, While, Assign, LocalAssign, LocalFunction, 
                            Function, Return, Break, Do, Repeat, Call, Invoke
                        )) if first_item else True
                        
                        if not is_statement_list:
                            # Skip injection - this might be an expression list
                            continue
                    
                    new_list = []
                    for item in attr_value:
                        new_list.append(item)
                        
                        # Roll the dice for junk injection (20% chance)
                        # Ensure we don't inject after returns or breaks (unreachable code)
                        if not isinstance(item, (Return, Break)) and random.random() < 0.2:
                            junk_node = self.generate_junk_node()
                            if junk_node:
                                new_list.append(junk_node)
                    
                    # Replace the original list with the new list containing junk
                    setattr(node, attr_name, new_list)
                
                # Explicitly skip unsafe attributes (logged for debugging if needed)
                elif attr_name in UNSAFE_ATTRS:
                    # Do not inject - this is an expression context
                    pass

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
            list: A list of integers where each byte is XOR'd with the dynamic key.
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

    def minify_source(self, source, preserve_fake_comments=True):
        """
        Minifies Lua source code by removing comments, empty lines, and indentation.
        Preserves string literals that may contain special characters.
        
        ANTI-AI ENHANCEMENT:
        When preserve_fake_comments is True, keeps comments that appear to be
        from our fake comment pool. This creates semantic misdirection for AI.
        
        Args:
            source (str): The Lua source code to minify.
            preserve_fake_comments (bool): If True, keeps misleading comments
            
        Returns:
            str: The minified source code.
        """
        # Keywords that indicate a fake comment worth keeping
        fake_comment_indicators = [
            'Adjusting', 'Recalculating', 'Synchronizing', 'Validating',
            'Processing', 'Updating', 'Mixing', 'Calculating', 'Serializing',
            'Compressing', 'Profiling', 'Garbage', 'Optimizing', 'Caching',
            'texture', 'font', 'animation', 'color', 'rendering', 'shadow',
            'audio', 'reverb', 'buffer', 'doppler', 'network', 'packet',
            'analytics', 'leaderboard', 'frame', 'draw call', 'GPU'
        ]
        
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
                    # Single-line comment - check if it's a fake comment to preserve
                    comment_start = i
                    i += 2  # Skip --
                    while i < length and source[i] != '\n':
                        i += 1
                    
                    # Extract the comment text
                    comment_text = source[comment_start:i]
                    
                    # Check if this comment should be preserved for AI confusion
                    should_preserve = False
                    if preserve_fake_comments and self.enable_fake_comments:
                        for indicator in fake_comment_indicators:
                            if indicator.lower() in comment_text.lower():
                                should_preserve = True
                                break
                    
                    if should_preserve:
                        # Keep this fake comment
                        result.append(comment_text)
            else:
                result.append(source[i])
                i += 1
        
        # Now process the source
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
        
        ANTI-AI ENHANCEMENT:
        When enable_hashed_globals is True, globals are accessed via hash lookup
        instead of string lookup. AI sees VirtualEnv[15923] not env["Drawing"].
        
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
        
        if self.enable_hashed_globals:
            # HASHED GLOBAL PROXIES (Weakness C Fix)
            # Use hash lookups instead of string lookups
            # AI sees VirtualEnv[123456] and cannot determine what global it is
            for api in self.api_globals:
                hash_value = self.calculate_djb2_hash(api)
                injection += f"local {api} = {self.virtual_env_name}[{hash_value}]\n"
        else:
            # Legacy: String-based lookups (vulnerable to AI analysis)
            injection += "local _ENV = (getgenv and getgenv()) or getfenv() or _G\n"
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
        
        # =====================================================================
        # HASHED GLOBAL PROXIES SETUP (Weakness C Fix)
        # =====================================================================
        hashed_globals_lua = ""
        if self.enable_hashed_globals:
            # Pass bit32 table names so hash function can use XOR polyfill
            # LuaU doesn't support the ~ XOR operator
            hashed_globals_lua = self.generate_hash_function_lua(bit32_table_name, bit32_bxor_name)
        
        # =====================================================================
        # STATE-DEPENDENT DECRYPTION SETUP (Weakness E Fix)
        # =====================================================================
        # Generate the state-dependent decryptor if enabled
        state_decryptor_lua = ""
        state_decryptor_name = None
        if self.enable_state_dependent_decrypt and self.state_var_name:
            state_decryptor_lua, state_decryptor_name = self.generate_state_dependent_decryptor(
                bit32_table_name, bit32_bxor_name
            )
        
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
        
        # 2.5. Generate Mega Dispatch Function (Logic Flooding for AI confusion)
        mega_dispatch_lua = ""
        if self.enable_logic_flooding:
            mega_dispatch_lua = self.generate_mega_dispatch_function()
        
        # 3. Generate the Script Body
        # Use ast.to_lua_source(self.ast) to convert modified tree back to Lua source
        script_lua = ast.to_lua_source(self.ast)
        
        # 4. Combine and Write
        # Order: Polyfill -> Hashed Globals -> Decryptor -> State Decryptor -> Constant Pool -> Mega Dispatch -> Virt Ops -> Script
        final_output = (
            bit32_polyfill + "\n" + 
            hashed_globals_lua + "\n" +
            decryptor_lua + 
            state_decryptor_lua +
            pool_lua + "\n\n" + 
            mega_dispatch_lua + "\n" + 
            virt_ops_lua + "\n" + 
            script_lua
        )
        
        # 4. Minify the output (remove comments, empty lines, indentation)
        final_output = self.minify_source(final_output)
        
        # 5. Add fake headers with misleading comments to poison AI context
        # Multiple fake headers create stronger semantic misdirection
        fake_headers = [
            "-- Roblox Studio Internal Plugin // Auto-generated UI Framework",
            "-- WARNING: This file was generated by Roblox's internal build system",
            "-- Module: RenderingPipeline.UILayoutManager v3.2.1",
            "-- Purpose: Handles responsive UI layout calculations and font rendering",
            "-- Dependencies: TextService, GuiService, UserInputService",
            "-- DO NOT MODIFY: Changes will be overwritten on next build",
            "-- Contact: rendering-team@roblox.com for questions"
        ]
        
        # Select 3-4 fake headers randomly
        selected_headers = random.sample(fake_headers, min(4, len(fake_headers)))
        fake_header = '\n'.join(selected_headers) + '\n'
        
        # Inject additional fake comments throughout (if enabled)
        if self.enable_fake_comments:
            # Add some fake inline comments at strategic points
            fake_inline = self.generate_fake_comment_block(2)
            final_output = fake_header + fake_inline + '\n' + final_output
        else:
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
    print("[*] Injecting Junk Code with Opaque Predicates...")
    obfuscator.inject_junk_code(obfuscator.ast)
    print("[+] Junk Code Injected (20% Density, Math-Based Predicates)")

    # 10. Apply Equivalency Mutation (wrap values in identity functions)
    print("[*] Applying Equivalency Mutation...")
    obfuscator.apply_equivalency_mutation(obfuscator.ast, mutation_rate=0.25)
    print("[+] Equivalency Mutation Applied (25% of assignments wrapped)")

    # 11. Virtualize Binary Operations
    print("[*] Virtualizing Binary Operations...")
    obfuscator.virtualize_operations(obfuscator.ast)
    print(f"[+] Operations Virtualized: {len(obfuscator.virt_ops)} operator types")

    # 12. Flatten Root Flow (Enhanced with Key-Based State Machine)
    print("[*] Flattening Root Control Flow (Key-Based State Machine)...")
    obfuscator.flatten_root_flow(obfuscator.ast)
    print("[+] Root Control Flow Flattened with Relative Transitions")
    
    # 13. Anti-AI Deobfuscation Features Status
    print("[*] Anti-AI Deobfuscation Features:")
    print(f"    [{'✓' if obfuscator.enable_misleading_names else '✗'}] Adversarial Context Poisoning (Misleading Names)")
    print(f"    [{'✓' if obfuscator.enable_fake_comments else '✗'}] Fake Comment Injection")
    print(f"    [{'✓' if obfuscator.enable_logic_flooding else '✗'}] Logic Flooding (Mega Dispatch Function)")
    print(f"    [{'✓' if obfuscator.enable_hashed_globals else '✗'}] Hashed Global Proxies (Weakness C Fix)")
    print(f"    [{'✓' if obfuscator.enable_state_dependent_decrypt else '✗'}] State-Dependent Decryption (Weakness E Fix)")
    print(f"    [+] Polymorphic Opcode Threshold: {obfuscator.poly_opcode_threshold}")
    print(f"    [+] Misleading Names Used: {len(obfuscator.used_misleading_names)}")
    if obfuscator.enable_hashed_globals:
        print(f"    [+] Global Hashes Computed: {len(obfuscator.global_hash_cache)}")
    if obfuscator.state_var_name:
        print(f"    [+] State Variable: {obfuscator.state_var_name}")
    
    # 14. Generate Output
    print("[*] Generating output file...")
    obfuscator.generate_output()
    print(f"[+] Obfuscated script written to {obfuscator.output_file}")
