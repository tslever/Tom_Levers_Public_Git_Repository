package neighbor_counting_utilities;


import java.util.Scanner;
import lattice_utilities.World;
import lattice_utilities.FarmingPattern;


public class IoManager {
	
	
	public static int[] heightWidthNumberOfSettlersAndRangeBasedOn(String[] args)
		throws AssertEqualsException,
			   NumberFormatException,
			   HeightOfWorldLessThanOneException,
			   WidthOfWorldLessThanOneException,
			   AreaOfWorldTooHighException,
			   UnrealisticNumberOfSettlersException,
			   RangeOfVonNeumannNeighborhoodLessThanZeroException {
		
		String usageMessage =
			"Arguments must be:\n" +
			"- Height of world, an integer, greater than 0;\n" +
			"- Width of world, an integer, greater than 0;\n" +
			"- Number of settlers, an integer, in the interval [0, width*height]; and\n" +
			"- Range of Von-Neumann Neighborhood, an integer, greater than or equal to 0.";
		
		int expectedNumberOfArguments = 4;
		
		assertEquals(usageMessage, expectedNumberOfArguments, args.length); // throws AssertionError.
		
		int[] heightWidthNumberOfSettlersAndRange = parsed(args); // throws NumberFormatException.
		
		checkAccordanceWithConstraintsOf(heightWidthNumberOfSettlersAndRange);
		
		return heightWidthNumberOfSettlersAndRange;
		
	}
	
	
	public static void assertEquals(String usageMessage, int expectedNumberOfArguments, int actualNumberOfArguments)
		throws AssertEqualsException {
		
		if (expectedNumberOfArguments != actualNumberOfArguments) {
			throw new AssertEqualsException(usageMessage);
		}
		
	}
	
	
	public static int[] parsed(String[] args) throws NumberFormatException {
		
		int[] argsAsIntegers = new int[args.length];
		
		for (int i = 0; i < args.length; i++) {
			
			argsAsIntegers[i] = Integer.parseInt(args[i]); // parseInt throws NumberFormatException.
			
		}
		
		return argsAsIntegers;
		
	}
	
	
	public static void checkAccordanceWithConstraintsOf(int[] heightWidthNumberOfSettlersAndRange)
		throws HeightOfWorldLessThanOneException,
			   WidthOfWorldLessThanOneException,
			   AreaOfWorldTooHighException,
			   UnrealisticNumberOfSettlersException,
			   RangeOfVonNeumannNeighborhoodLessThanZeroException {
		
		int heightOfWorld = heightWidthNumberOfSettlersAndRange[0];
		int widthOfWorld = heightWidthNumberOfSettlersAndRange[1];
		int numberOfSettlers = heightWidthNumberOfSettlersAndRange[2];
		
		if (heightOfWorld < 1) {
			throw new HeightOfWorldLessThanOneException("Height of world must be greater than 0.");
		}
		
		if (widthOfWorld < 1) {
			throw new WidthOfWorldLessThanOneException("Width of world must be greater than 0.");
		}
		
		// heightOfWorld * widthOfWorld must be less than 2147483647 to prevent integer overflow.
		// Given heightOfWorld, widthOfWorld must be less than or equal to floor( 2147483647 / heightOfWorld ).
		// Given widthOfWorld, heightOfWorld must be less than or equal to floor( 2147483647 / widthOfWorld ).
		int max_int = 2147483647;
		if ((widthOfWorld > max_int / heightOfWorld) || (heightOfWorld > max_int / widthOfWorld)) {
			throw new AreaOfWorldTooHighException(
				"The area of the world must be less than or equal to max_int = 2,147,483,647.");
		}
		
		
		if ((numberOfSettlers < 0) || (numberOfSettlers > heightOfWorld * widthOfWorld)) {
			throw new UnrealisticNumberOfSettlersException(
				"Number of settlers must be in the interval [0, width*height].");
		}
		
		if (heightWidthNumberOfSettlersAndRange[3] < 0) {
			throw new RangeOfVonNeumannNeighborhoodLessThanZeroException(
				"Range of von-Neumann neighborhood must be greater than or equal to 0.");
		}
		
	}
	
	
	static Scanner scanner = new Scanner(System.in);
	
	
	public static void displayIntroduction(
		int heightOfWorld, int widthOfWorld, int numberOfSettlers, int rangeOfVonNeumannNeighborhood) {
		
		System.out.println(
			"You seek to count neighbors in von-Neumann neighborhoods when\n" +
			"von-Neumann neighborhoods are placed in a two-dimensional array of cells.\n" +
			"The array has a height of " + heightOfWorld + " cells and a width of " + widthOfWorld + " cells.\n" +
			"A node is a randomly selected cell at which a von-Neumann neighborhood will be centered.\n" +
			"There are " + numberOfSettlers + " nodes and von-Neumann neighborhoods.\n" +
			"von-Neumann neighborhoods may blend together.\n" +
			"Each von-Neumann neighborhood has a range of " + rangeOfVonNeumannNeighborhood + ".\n\n" +
			
			"Alternatively:\n" +
			"You seek to count cells settled or farmed in a world of cells with\n" +
			"a height of " + heightOfWorld + " cells and a width of " + widthOfWorld + " cells.\n" +
			"The world is settled by " + numberOfSettlers + " settlers, each of whom chooses a random location.\n" +
			"Each settler farms the land around him or her in a von-Neumann pattern of range " +
			rangeOfVonNeumannNeighborhood + ".\n\n"
		);
		
	}
	
	
	public static void askAboutAndPossiblyDisplaySettledWorld(World world) {
		
		String answer = "";
		
		while (!answer.equals("y") && !answer.equals("n")) {
			
			System.out.print("Display array of nodes / settled world (y/n)? ");
		
			answer = scanner.nextLine();
			
		}
		
		if (answer.equals("y")) {

			System.out.println("\nSettled World:");

			world.display();
			
		}
		
		System.out.println();
		
	}
	
	
	public static void askAboutAndPossiblyDisplayFarmingPattern(FarmingPattern farmingPattern) {
		
		String answer = "";
		
		while (!answer.equals("y") && !answer.equals("n")) {
			
			System.out.print("Display von-Neumann neighborhood / farming pattern (y/n)? ");
		
			answer = scanner.nextLine();
			
		}
		
		if (answer.equals("y")) {

			System.out.println("\nFarming Pattern:");

			farmingPattern.display();
			
		}
		
		System.out.println();
		
	}
	
	
	public static void askAboutAndPossiblyDisplaySettledAndFarmedWorld(World world) {
		
		String answer = "";
		
		while (!answer.equals("y") && !answer.equals("n")) {
			
			System.out.print("Display array with von-Neumann neighborhoods / settled and farmed world (y/n)? ");
		
			answer = scanner.nextLine();
			
		}
		
		if (answer.equals("y")) {

			System.out.println("\nSettled and Farmed World:");

			world.display();
			
		}
		
		System.out.println();
		
	}
	
	
}
