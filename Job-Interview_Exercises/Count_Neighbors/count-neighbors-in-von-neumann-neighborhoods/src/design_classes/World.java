package design_classes;


/**
 * The World is a singleton.
 * The instance of the World:
 * - Resembles a sheet of paper.
 * - Has a width and a height.
 *   The width and the height are specified as command-line arguments.
 *   If the World has a width less than 3, then the World wraps horizontally
 *   (i.e., one vertical edge of the World is colinear with the other vertical edge of the World).
 *   If the World has a width greater than or equal to 3, then the World does not wrap horizontally.
 *   If the World has a height less than 3, then the World wraps vertically
 *   (i.e., one horizontal edge of the World is colinear with the other horizontal edge of the World).
 *   If the World has a height greater than or equal to 3, then the World does not wrap vertically.
 * - Consists of an identically sized two-dimensional array of cells.
 *   Each cell may become imprinted with one settler (i.e., a 1).
 *   The desired number of settlers in the World is specified as a command-line argument.
 * - The imprint of a von-Neumann neighborhood will be centered around each settler.
 *   von-Neumann neighborhoods may overlap.
 *   If a von-Neumann neighborhood were to extend beyond a lone edge of the World, the neighborhood is truncated.
 * - Each cell within a von-Neumann neighborhood, or the intersection of multiple neighborhoods,
 *   will become imprinted with one settler.
 * 
 * @author Tom Lever
 *
 */
public class World {

}