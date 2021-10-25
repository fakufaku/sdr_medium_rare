# Copyright 2021 Robin Scheibler
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import collections
from typing import Optional, Tuple

import torch as pt

from .dtypes import dtype_cpx2f, dtype_f2cpx, is_complex_type

complex_types = [pt.complex64, pt.complex128]


def divide(num, denom, eps=1e-7):
    if num.dtype in complex_types and denom.dtype not in complex_types:
        return pt.view_as_complex(
            pt.view_as_real(num) / pt.clamp(denom[..., None], min=eps)
        )
    elif num.dtype not in complex_types and denom.dtype not in complex_types:
        return num / pt.clamp(denom, min=eps)
    elif num.dtype in complex_types and denom.dtype in complex_types:
        return num / denom
    else:
        raise NotImplementedError(
            f"Division of {num.dtype} by {denom.dtype} is not implemented"
        )


def multiply(tensor1: pt.Tensor, tensor2: pt.Tensor):

    if pt.is_complex(tensor1) and pt.is_complex(tensor2):
        # both tensors are complex
        return tensor1 * tensor2

    elif not pt.is_complex(tensor1) and not pt.is_complex(tensor2):
        # both tensors are real
        return tensor1 * tensor2

    else:
        # one is real, one is complex
        if pt.is_complex(tensor1):
            tensor1, tensor2 = tensor2, tensor1

        # tensor1 is real, tensor2 is complex
        return pt.view_as_complex(pt.view_as_real(tensor2) * tensor1[..., None])


def mag_sq(x: pt.Tensor):
    if x.dtype in [pt.complex64, pt.complex128]:
        return x.real.square() + x.imag.square()
    else:
        return x.square()


def mag(x: pt.Tensor):
    return pt.sqrt(mag_sq(x))


def complex_2_real(A: pt.Tensor, vector=False) -> pt.Tensor:
    """
    Create the real matrix equivalent to the complex input A
    """
    shape = A.shape[:-2]
    n_rows, n_cols = A.shape[-2:]

    if vector:
        shape_A_ri = shape + (2 * n_rows, n_cols)
    else:
        shape_A_ri = shape + (2 * n_rows, 2 * n_cols)

    A_ri = A.new_zeros(shape_A_ri, dtype=dtype_cpx2f(A))
    A_ri[..., :n_rows, :n_cols] = A.real
    A_ri[..., n_rows:, :n_cols] = A.imag

    if not vector:
        A_ri[..., :n_rows, n_cols:] = -A.imag
        A_ri[..., n_rows:, n_cols:] = A.real

    return A_ri


def bmm(input: pt.Tensor, mat2: pt.Tensor) -> pt.Tensor:
    m1 = pt.view_as_real(input)
    m2 = pt.view_as_real(mat2)
    real_part = m1[..., 0] @ m2[..., 0] - m1[..., 1] @ m2[..., 1]
    imag_part = m1[..., 0] @ m2[..., 1] + m1[..., 1] @ m2[..., 0]
    out = pt.cat((real_part[..., None], imag_part[..., None]), dim=-1)
    out = pt.view_as_complex(out)
    return out


def hermite(A: pt.Tensor, dim1: Optional[int] = -2, dim2: Optional[int] = -1):
    if A.dtype in complex_types:
        return pt.conj(A.transpose(dim1, dim2))
    else:
        return A.transpose(dim1, dim2)


def solve_loaded(A: pt.Tensor, b: pt.Tensor, load=1e-6):
    eye = pt.eye(A.shape[-1]).type_as(A)
    load_factor = (
        load * A.abs().max(dim=-2, keepdim=True).values.max(dim=-1, keepdim=True).values
    )
    load_factor = pt.clamp(load_factor, min=load)
    return pt.linalg.solve(A + load_factor * eye, b)


def inv_loaded(A: pt.Tensor, load=1e-6):
    eye = pt.eye(A.shape[-1]).type_as(A)
    load_factor = (
        load * A.abs().max(dim=-2, keepdim=True).values.max(dim=-1, keepdim=True).values
    )
    load_factor = pt.clamp(load_factor, min=load)
    return pt.linalg.inv(A + load_factor * eye)


def solve(
    b: pt.Tensor, A: pt.Tensor, out: Optional[pt.Tensor] = None
) -> Tuple[pt.Tensor, pt.Tensor]:

    if not is_complex_type(b) and not is_complex_type(A):
        return pt.solve(b, A)

    shape = A.shape[:-2]
    n_dim = A.shape[-2]

    # transform from complex to equivalent real format
    A_ri = complex_2_real(A)
    b_ri = complex_2_real(b, vector=True)

    x_ri, _ = pt.solve(b_ri, A_ri)

    shape_out = shape + (n_dim, b.shape[-1])
    if out is None:
        out = x_ri.new_zeros(shape_out, dtype=dtype_f2cpx(x_ri))
    else:
        assert out.shape == shape_out
        assert is_complex_type(out)

    out_r = pt.view_as_real(out)
    out_r[:, :, :, 0] = x_ri[:, :n_dim, :]
    out_r[:, :, :, 1] = x_ri[:, n_dim:, :]

    # we return a None to match the interface of torch.solve
    return out, None


def diagonal_loading(A: pt.Tensor, d: pt.Tensor):
    """
    Load the diagonal of A with the vector d
    """
    D = pt.diag(pt.arange(A.shape[-1], device=A.device) * eps)
    D = D.reshape(pt.Size([1] * (len(A.shape) - 2)) + A.shape[-2:])
    return A + D


def eigh_2x2(
    A: pt.Tensor, B: Optional[pt.Tensor] = None, eps: Optional[float] = 0.0
) -> Tuple[pt.Tensor, pt.Tensor]:
    """
    Specialized routine for batched 2x2 EVD and GEVD for complex hermitian matrices
    """
    assert 2 == A.shape[-1]
    assert A.shape[-2] == A.shape[-1]

    if B is not None:
        assert B.shape[-1] == 2
        assert B.shape[-1] == B.shape[-2]

        # broadcast
        A, B = pt.broadcast_tensors(A, B)

        # notation
        a11 = A[..., 0, 0]
        a12 = A[..., 0, 1]
        a22 = A[..., 1, 1]
        b11 = B[..., 0, 0]
        b12 = B[..., 0, 1]
        b22 = B[..., 1, 1]

        # coefficient of secular equation: x - b * x + c
        a11b22 = a11.real * b22.real
        a22b11 = a22.real * b11.real
        re_a12b12c = a12.real * b12.real + a12.imag * b12.imag
        b = a11b22 + a22b11 - 2.0 * re_a12b12c

        det_A = a11.real * a22.real - mag_sq(a12)
        det_B = b11.real * b22.real - mag_sq(b12)
        c = det_A * det_B

        # discrimant of secular equation
        delta = pt.square(b) - 4 * c

        # we clamp to zero to avoid numerical inaccuracies
        # we know the minimum is zero because A and B should
        # be symmetric or hermitian symmetric
        delta = pt.clamp(delta, min=eps)

        # fill the eigenvectors in ascending order
        eigenvalues = pt.zeros(A.shape[:-1], device=A.device)
        eigenvalues[..., 0] = 0.5 * (b - pt.sqrt(delta))  # small eigenvalue
        eigenvalues[..., 1] = 0.5 * (b + pt.sqrt(delta))  # large eigenvalue

        # normalize the eigenvalues
        eigenvalues = eigenvalues / pt.clamp(det_B[..., None], min=eps)

        # notation
        ev1 = eigenvalues[..., 0]
        ev2 = eigenvalues[..., 1]

        # now fill the eigenvectors
        eigenvectors = A.new_zeros(A.shape)
        # vector corresponding to small eigenvalue
        eigenvectors[..., 0, 0] = multiply(ev1, b12) - a12
        eigenvectors[..., 1, 0] = a11 - ev1 * b11
        # vector corresponding to large eigenvalue
        eigenvectors[..., 0, 1] = multiply(ev2, b12) - a12
        eigenvectors[..., 1, 1] = a11 - ev2 * b11

    else:
        # Do the EVD

        # secular equation: a * lambda^2 - b * lambda + c
        # where lambda is the eigenvalue
        b = A[..., 0, 0].real + A[..., 1, 1].real
        c = A[..., 0, 0].real * A[..., 1, 1].real - mag_sq(A[..., 0, 1])

        # discrimant of secular equation
        delta = pt.square(b) - 4 * c

        # we clamp to zero to avoid numerical inaccuracies
        # we know the minimum is zero because A and B should
        # be symmetric or hermitian symmetric
        delta = pt.clamp(delta, min=eps)

        # fill the eigenvectors in ascending order
        eigenvalues = delta.new_zeros(A.shape[:-1])
        eigenvalues[..., 0] = 0.5 * (delta - pt.sqrt(delta))  # small eigenvalue
        eigenvalues[..., 1] = 0.5 * (delta + pt.sqrt(delta))  # large eigenvalue

        # now fill the eigenvectors
        eigenvectors = A.new_zeros(A.shape)
        # vector corresponding to small eigenvalue
        eigenvectors[..., 0, 0] = A[..., 0, 1]
        eigenvectors[..., 1, 0] = eigenvalues[..., 0] - A[..., 1, 1]
        # vector corresponding to large eigenvalue
        eigenvectors[..., 0, 1] = eigenvalues[..., 1] - A[..., 0, 0]
        eigenvectors[..., 1, 1] = pt.conj(eigenvectors[..., 0, 0])

    norm = pt.sqrt(pt.sum(mag_sq(eigenvectors), dim=-2, keepdim=True))
    eigenvectors = divide(eigenvectors, norm, eps=eps)

    return eigenvalues, eigenvectors


def eigh_wrapper(V, use_cpu=True):

    if use_cpu:
        dev = V.device

        # diagonal loading factor
        # dload = 1e-5 * pt.eye(V.shape[-1], device=V.device, dtype=V.dtype)
        # V = (V + dload).cpu()
        V = V.cpu()
        e_val, e_vec = pt.linalg.eigh(V)

        return e_val.to(dev), e_vec.to(dev)
    else:
        return pt.linalg.eigh(V)


def eigh(
    A: pt.Tensor,
    B: Optional[pt.Tensor] = None,
    eps: Optional[float] = 1e-15,
    use_eigh_cpu: Optional[bool] = True,
) -> Tuple[pt.Tensor, pt.Tensor]:
    """
    Eigenvalue decomposition of a complex Hermitian symmetric matrix
    """

    """
    # for 2x2, use specialized routine
    if A.shape[-1] == 2:
        return eigh_2x2(A, B=B, eps=eps)
    """

    """
    # here we add a small diagonal loading to make sure that all eigenvalues
    # are distinct, otherwise, the algorithm may fail sometimes
    D = pt.diag(pt.arange(A.shape[-1], device=A.device) * eps)
    D = D.reshape(pt.Size([1] * (len(A.shape) - 2)) + A.shape[-2:])
    A = A + D
    """

    if B is not None:
        assert A.shape == B.shape, "A and B should have the same size"

        # generalized eigenvalue decomposition
        Bval, Bvec = eigh_wrapper(B, use_cpu=use_eigh_cpu)
        Bval_sqrt = pt.sqrt(pt.clamp(Bval, min=1e-5))
        B_sqrt_inv = Bvec @ (pt.reciprocal(Bval_sqrt[..., :, None]) * hermite(Bvec))

        Linv_A_LHinv = B_sqrt_inv @ A @ hermite(B_sqrt_inv)

        eigenvalues, eigenvectors = eigh_wrapper(Linv_A_LHinv, use_cpu=use_eigh_cpu)

        eigenvectors = hermite(B_sqrt_inv) @ eigenvectors

        # normalize the eigenvectors
        norm = pt.linalg.norm(eigenvectors, dim=-2, keepdim=True)
        eigenvectors = eigenvectors / pt.clamp(norm, min=1e-5)

    else:
        # regular eigenvalue decomposition
        eigenvalues, eigenvectors = eigh_wrapper(A, use_cpu=use_eigh_cpu)

    return_type_eigh = collections.namedtuple(
        "return_type_torchiva_linalg_eigh", ("eigenvalues", "eigenvectors")
    )

    return return_type_eigh(eigenvalues, eigenvectors)
