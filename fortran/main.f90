program main
  use solver_mod
  use test_mod
  implicit none

  integer, parameter :: n = 8192
  integer, parameter :: cg_max_iter = 32
  logical, parameter :: RANDOMISE_SEED = .true.

  real, allocatable :: A(:,:)
  real, allocatable :: x_soln(:), b(:), x(:)
  integer :: iters
  real :: av_error2
  integer :: i

  ! For timing
  integer :: count1, count2, count_rate
  integer(8) :: duration_us

  if (.not. run_tests()) then
    print *, "A test failed!"
    stop 1
  end if

  allocate(A(n, n))
  allocate(x_soln(n))
  allocate(b(n))
  allocate(x(n))

  ! Initial conditions
  call generate_positive_definite(A, n)
  print *, "Generating random solution"
  call fill_rand_vec(x_soln, n, -1.0, 1.0)
  print *, "Generating right hand side"
  call calc_b(b, A, x_soln, n)
  
  x = 0.0

  ! Solve
  call system_clock(count1, count_rate)
  iters = cg_solve(x, A, b, n, cg_max_iter)
  call system_clock(count2)
  
  duration_us = (int(count2 - count1, 8) * 1000000_8) / int(count_rate, 8)

  print *, "Performed ", iters, " iterations"
  print *, "Solve time: ", duration_us, " us"
  if (iters > 0) print *, "Time per iteration: ", duration_us / iters, " us"

  av_error2 = 0.0
  do i = 1, n
    av_error2 = av_error2 + abs(x_soln(i) - x(i))
  end do
  av_error2 = av_error2 / n

  print *, "Average error = ", sqrt(av_error2)

  deallocate(A)
  deallocate(x_soln)
  deallocate(b)
  deallocate(x)

contains

  function run_tests() result(all_passed)
    logical :: all_passed
    all_passed = .true.
    
    if (.not. test_matvec_identity()) then
      all_passed = .false.
      print *, "matvec identity failed!"
    end if

    if (.not. test_matvec_simple()) then
      all_passed = .false.
      print *, "matvec simple failed!"
    end if

    if (.not. test_dot()) then
      all_passed = .false.
      print *, "dot failed!"
    end if
  end function run_tests

  subroutine calc_b(b, A, x, n)
    integer, intent(in) :: n
    real, intent(out) :: b(n)
    real, intent(in) :: A(n,n), x(n)
    integer :: i, j

    !$omp parallel do
    do i = 1, n
      b(i) = 0.0
      do j = 1, n
        b(i) = b(i) + A(i, j) * x(j)
      end do
    end do
  end subroutine calc_b

  subroutine fill_rand_vec(x, size, min_val, max_val)
    integer, intent(in) :: size
    real, intent(out) :: x(size)
    real, intent(in) :: min_val, max_val
    integer :: i
    real :: r

    ! In Fortran, random_seed / random_number is standard
    ! For simplicity and thread safety, let's just initialize the seed once
    ! in a simple way or let it be.
    ! Here we just use random_number array assignment which handles the whole array
    call random_number(x)
    !$omp parallel do
    do i = 1, size
      x(i) = min_val + x(i) * (max_val - min_val)
    end do
  end subroutine fill_rand_vec

  subroutine generate_positive_definite(A, n)
    integer, intent(in) :: n
    real, intent(out) :: A(n,n)
    real, allocatable :: B(:,:)
    integer :: i, j

    print *, "Generating matrix"
    allocate(B(n,n))
    call random_number(B)

    ! Make a symmetric matrix
    !$omp parallel do private(j)
    do i = 1, n
      do j = 1, n
        A(i, j) = (B(i, j) + B(j, i)) / 2.0
      end do
    end do

    ! Add n * identity to make diagonally dominant => positive definite
    do i = 1, n
      A(i, i) = A(i, i) + max(real(n) / 32.0, 1.0)
    end do

    deallocate(B)
  end subroutine generate_positive_definite

end program main