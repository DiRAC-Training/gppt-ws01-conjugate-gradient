module solver_mod
  implicit none

contains

  ! Dense matrix-vector product: y = A * x.
  subroutine matvec(y, A, x, n)
    integer, intent(in) :: n
    real, intent(out) :: y(n)
    real, intent(in) :: A(n,n)
    real, intent(in) :: x(n)
    integer :: i, j
    real :: sum_val

    !$omp parallel do private(j, sum_val)
    do i = 1, n
      sum_val = 0.0
      do j = 1, n
        sum_val = sum_val + A(i, j) * x(j) 
      end do
      y(i) = sum_val
    end do
  end subroutine matvec

  ! Dot product: result = sum(a[i] * b[i]).
  function dot(a, b, n) result(sum_val)
    integer, intent(in) :: n
    real, intent(in) :: a(n)
    real, intent(in) :: b(n)
    real :: sum_val
    integer :: i

    sum_val = 0.0
    !$omp parallel do reduction(+:sum_val)
    do i = 1, n
      sum_val = sum_val + a(i) * b(i)
    end do
  end function dot

  ! AXPBY operation: y = alpha * x + beta * y.
  subroutine axpby(y, x, alpha, beta, n)
    integer, intent(in) :: n
    real, intent(inout) :: y(n)
    real, intent(in) :: x(n)
    real, intent(in) :: alpha
    real, intent(in) :: beta
    integer :: i

    !$omp parallel do
    do i = 1, n
      y(i) = alpha * x(i) + beta * y(i)
    end do
  end subroutine axpby

  ! Solve A*x = b using the conjugate gradient method.
  function cg_solve(x, A, b, n, max_iter) result(n_iter)
    integer, intent(in) :: n, max_iter
    real, intent(inout) :: x(n)
    real, intent(in) :: A(n,n)
    real, intent(in) :: b(n)
    integer :: n_iter
    
    real, allocatable :: r(:), p(:), A_times_p(:)
    real :: residual_sq_old, residual_sq_new
    real :: alpha, beta
    integer :: i
    real, parameter :: EPS = epsilon(1.0)

    allocate(r(n))
    allocate(p(n))
    allocate(A_times_p(n))

    ! Step 1: r_0 = f - K*x_0
    call matvec(r, A, x, n)
    !$omp parallel do
    do i = 1, n
      r(i) = b(i) - r(i)
    end do

    ! Step 2: p_0 = r_0
    p = r

    residual_sq_old = dot(r, r, n)

    do n_iter = 1, max_iter
      ! Step 3a: alpha_k = (r_k . r_k) / (p_k . K*p_k)
      call matvec(A_times_p, A, p, n)
      alpha = residual_sq_old / dot(p, A_times_p, n)

      ! Step 3b: x_{k+1} = x_k + alpha_k * p_k
      call axpby(x, p, alpha, 1.0, n)

      ! Step 3c: r_{k+1} = r_k - alpha_k * K*p_k
      call axpby(r, A_times_p, -alpha, 1.0, n)

      ! Step 3d: beta_k = (r_{k+1} . r_{k+1}) / (r_k . r_k)
      !          p_{k+1} = r_{k+1} + beta_k * p_k
      residual_sq_new = dot(r, r, n)
      
      ! This method is so good it crashes if the residual gets too small!
      if (residual_sq_new < EPS) exit
      
      beta = residual_sq_new / residual_sq_old
      call axpby(p, r, 1.0, beta, n)
      residual_sq_old = residual_sq_new

      print '(I0, A, E15.6)', n_iter, ': r = ', residual_sq_new / n
    end do

    deallocate(r)
    deallocate(p)
    deallocate(A_times_p)
  end function cg_solve

end module solver_mod
