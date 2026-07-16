module test_mod
  use solver_mod
  use precision_mod
  implicit none

contains

  ! Test matvec with an identity matrix
  function test_matvec_identity() result(passed)
    logical :: passed
    integer, parameter :: n = 3
    real(wp) :: A(n*n)
    real(wp) :: x(n), y(n)
    integer :: i

    A = 0.0_wp
    do i = 1, n
      A(idx(i, i, n)) = 1.0_wp
      x(i) = i
    end do

    !$omp target data map(to: A(1:n*n), x(1:n)) map(from: y(1:n))
    call matvec(y, A, x, n)
    !$omp end target data

    passed = .true.
    do i = 1, n
      if (abs(y(i) - x(i)) > epsilon(1.0_wp)) passed = .false.
    end do
  end function test_matvec_identity

  function test_matvec_simple() result(passed)
    logical :: passed
    integer, parameter :: n = 3
    real(wp) :: mat(n*n)
    real(wp) :: x(n), y(n), y_soln(n)
    integer :: i

    ! Create a matrix with known values
    ! Assign via idx so the layout follows the solver (row- or column-major).
    mat(idx(1, 1, n)) = -1.0_wp; mat(idx(1, 2, n)) = -6.0_wp;   mat(idx(1, 3, n)) =   2.0_wp
    mat(idx(2, 1, n)) =  4.0_wp; mat(idx(2, 2, n)) =  3.0_wp;   mat(idx(2, 3, n)) =  10.0_wp
    mat(idx(3, 1, n)) =  0.0_wp; mat(idx(3, 2, n)) = -100.0_wp; mat(idx(3, 3, n)) =   1.0_wp
    
    ! x and y_soln are calculated solutions to y = Ax.
    x = [-1.0, 2.0, 0.0]

    ! M * x =
    ! -1*-1 + -6*2 +  2*0 =  -11
    !  4*-1 +  3*2 + 10*0 =    2
    !  0*-1 + -100*2 + 1*0 = -200
    y_soln = [-11.0, 2.0, -200.0]

    !$omp target data map(to: mat(1:n*n), x(1:n)) map(from: y(1:n))
    call matvec(y, mat, x, n)
    !$omp end target data

    passed = .true.
    do i = 1, n
      if (abs(y(i) - y_soln(i)) > epsilon(1.0_wp) * 10) passed = .false.
    end do
  end function test_matvec_simple

  function test_dot() result(passed)
    logical :: passed
    integer, parameter :: n = 1024
    real(wp) :: x(n), y(n)
    real(wp) :: res, soln, p
    integer :: i

    ! Generate x and y and calculate their dot product in this loop
    soln = 0.0_wp
    do i = 1, n
      p = real(i - 1, wp) / real(n, wp)
      x(i) = p
      y(i) = p
      soln = soln + p * p
    end do

    ! Calculate dot with the function and test against above value
    !$omp target data map(to: x(1:n), y(1:n))
    res = dot(x, y, n)
    !$omp end target data
    passed = abs(res - soln) < 1.0e-3_wp
  end function test_dot

end module test_mod
