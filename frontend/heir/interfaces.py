from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class EncValue:
  """Encrypted/Encoded Value produced by a backend."""

  # Identifier that enables the backend to reject mismatches
  # E.g., cannot swap x=enc_a(..) and y=enc_b(..) when calling eval
  identifier: str

  # The ciphertext
  value: object


# Type Hint for Encryption Functions
EncFunc = Callable[..., EncValue]


@dataclass
class CompilationResult:
  # The module object containing the compiled functions
  module: object

  # The function name used to generate the various compiled functions
  func_name: str

  # A list of arg names (in order)
  arg_names: list[str]

  # A list of indices of secret args
  secret_args: list[int]

  # A mapping from argument name to the compiled encryption function
  arg_enc_funcs: Optional[dict[str, EncFunc]] = None

  # The compiled decryption function for the function result
  result_dec_func: Optional[Callable] = None

  # The main compiled function
  main_func: Optional[Callable] = None

  # Backend setup functions, if any
  setup_funcs: Optional[dict[str, Callable]] = None


class ClientInterface(ABC):

  @abstractmethod
  def setup(self):
    """Configure the initial cryptosystem and this wrapper interface."""
    ...

  @abstractmethod
  def decrypt_result(self, result, **kwargs):
    ...

  @abstractmethod
  def __getattr__(self, key):
    """Invoke a function with a dynamically-generated name:

    - encrypt_{arg_name}
    - {func_name}
    """
    ...

  @abstractmethod
  def __call__(self, *args, **kwargs):
    """Invoke setup, encryption, eval and decryption seamlessly."""
    ...

  def original(self, *args, **kwargs):
    """Forwards to the original function."""
    if self.func is None:
      raise ValueError("Function not initialized")
    return self.func(*args, **kwargs)


class BackendInterface(ABC):

  @abstractmethod
  def run_backend(*args, **kwargs) -> ClientInterface:
    ...
