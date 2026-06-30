module idx_mod
  implicit none
contains
  pure integer function idx(i, j, n)
    integer, intent(in) :: i, j, n
    ! In C++, idx = i * n + j (0-indexed). 
    ! In Fortran, 2D arrays are column major.
    ! We will just use standard Fortran 2D arrays, 
    ! so we don't necessarily need this function for 2D arrays,
    ! but if we use 1D arrays to mimic C++:
    ! idx = i * n + j + 1 ! 1-indexed
    idx = i * n + j + 1
  end function idx
end module idx_mod