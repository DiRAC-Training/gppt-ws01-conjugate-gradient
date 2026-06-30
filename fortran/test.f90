module test_mod
  use solver_mod
  implicit none

contains

  function test_matvec_identity() result(passed)
    logical :: passed
    integer, parameter :: n = 3
    real :: A(n,n)
    real :: x(n), y(n)
    integer :: i, j

    ! A = Identity
    A = 0.0
    do i = 1, n
      A(i,i) = 1.0
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
    real :: A(n,n)
    real :: x(n), y(n), expected(n)
    integer :: i

    A = reshape([1.0, 4.0, 7.0, &
                 2.0, 5.0, 8.0, &
                 3.0, 6.0, 9.0], [3, 3]) ! Fortran reshape fills column-major

    x = [1.0, 2.0, 3.0]
    
    ! A * x =
    ! 1*1 + 2*2 + 3*3 = 14
    ! 4*1 + 5*2 + 6*3 = 32
    ! 7*1 + 8*2 + 9*3 = 50
    expected = [14.0, 32.0, 50.0]

    call matvec(y, A, x, n)

    passed = .true.
    do i = 1, n
      if (abs(y(i) - expected(i)) > epsilon(1.0) * 10) passed = .false.
    end do
  end function test_matvec_simple

  function test_dot() result(passed)
    logical :: passed
    integer, parameter :: n = 3
    real :: x(n), y(n)
    real :: result

    x = [1.0, 2.0, 3.0]
    y = [4.0, 5.0, 6.0]
    
    ! 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    result = dot(x, y, n)

    passed = abs(result - 32.0) < epsilon(1.0) * 10
  end function test_dot

end module test_mod