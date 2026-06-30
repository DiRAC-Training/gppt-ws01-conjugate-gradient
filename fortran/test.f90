module test_mod
  use solver_mod
  implicit none

contains

  function test_matvec_identity() result(passed)
    logical :: passed
    integer, parameter :: n = 3
    real :: A(n*n)
    real :: x(n), y(n)
    integer :: i

    ! A = Identity
    A = 0.0
    do i = 1, n
      A(idx(i, i, n)) = 1.0
      x(i) = i
    end do

    call matvec(y, A, x, n)

    passed = .true.
    do i = 1, n
      if (abs(y(i) - x(i)) > epsilon(1.0)) passed = .false.
    end do
  end function test_matvec_identity

  function test_matvec_simple() result(passed)
    logical :: passed
    integer, parameter :: n = 3
    real :: A(n*n)
    real :: x(n), y(n), expected(n)
    integer :: i

    ! Non-symmetric matrix (flat row-major, same layout/values as the C++ test):
    ! M = [-1 -6  2; 4 3 10; 0 -100 1]
    A = [-1.0, -6.0,   2.0, &
          4.0,  3.0,  10.0, &
          0.0, -100.0, 1.0]

    x = [-1.0, 2.0, 0.0]

    ! M * x =
    ! -1*-1 + -6*2 +  2*0 =  -11
    !  4*-1 +  3*2 + 10*0 =    2
    !  0*-1 + -100*2 + 1*0 = -200
    expected = [-11.0, 2.0, -200.0]

    call matvec(y, A, x, n)

    passed = .true.
    do i = 1, n
      if (abs(y(i) - expected(i)) > epsilon(1.0) * 10) passed = .false.
    end do
  end function test_matvec_simple

  function test_dot() result(passed)
    logical :: passed
    integer, parameter :: n = 1024
    real :: x(n), y(n)
    real :: res, soln, p
    integer :: i

    ! Generate x and y as a ramp p = i/n and accumulate the reference dot product.
    ! (i-1 keeps the 0-based indexing of the C++ test: p = float(i)/float(n).)
    soln = 0.0
    do i = 1, n
      p = real(i - 1) / real(n)
      x(i) = p
      y(i) = p
      soln = soln + p * p
    end do

    res = dot(x, y, n)
    passed = abs(res - soln) < 1.0e-3
  end function test_dot

end module test_mod