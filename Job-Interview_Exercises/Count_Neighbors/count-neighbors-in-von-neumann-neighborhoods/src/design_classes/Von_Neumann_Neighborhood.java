package design_classes;


/**
 * The instance of the Von_Neumann_Neighborhood:
 * - Will have a range (0 <= r < infinity) specified as a command-line argument.
 * - Consists of a two-dimensional array with width and height equal to (1 + 2*r).
 *   The array will be initialized with 0's.
 *   r rows will be above the center row; r rows will be below the center row.
 *   r columns will be to the left of the center column; r columns will be to the right of the center column.
 *   The array will be imprinted with 1's with, for large r:
 *   - (1+2*0) 1 at (row 0, columns [r-0, r+0]).
 *   - (1+2*1) 1's at (row 1, columns [r-1, r+1]).
 *   - (1+2*2) 1's at (row 2, columns [r-2, r+2]).
 *   - ...
 *   - (1+2*(r-1)) 1's at (row r-1, columns [r-(r-1), r+(r-1)]).
 *   - (1+2*r) 1's at (row r, columns [r-r, r+r]).
 *   - (1+2*(r-1)) 1's at (row r+1, columns [r-(r-1), r+(r-1)]).
 *   - ...
 *   - (1+2*2) 1's at (row 1+2*r-3, columns [r-2, r+2]).
 *   - (1+2*1) 1's at (row 1+2*r-2, columns [r-1, r+1]).
 *   - (1+2*0) 1's at (row 1+2*r-1, columns [r-0, r+0]).
 * 
 * @author Tom Lever
 *
 */
public class Von_Neumann_Neighborhood extends FarmingPattern {

}