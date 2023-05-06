package neighbor_counting_utilities;


import lattice_utilities.VonNeumannNeighborhood;
import lattice_utilities.EuclidNeighborhood;
import lattice_utilities.World;
import settler_utilities.Settlers;


public class NeighborCounter {

	public static void main(String[] args)
		throws AssertEqualsException,
		   NumberFormatException,
		   HeightOfWorldLessThanOneException,
		   WidthOfWorldLessThanOneException,
		   AreaOfWorldTooHighException,
		   UnrealisticNumberOfSettlersException,
		   RangeOfVonNeumannNeighborhoodLessThanZeroException {

		Data data = new Data(IoManager.heightWidthNumberOfSettlersAndRangeBasedOn(args));
		
		IoManager.displayIntroduction(
			data.heightOfWorld(), data.widthOfWorld(), data.numberOfSettlers(), data.rangeOfFarmingPattern());
		
		World world = new World(data.heightOfWorld(), data.widthOfWorld());
		
		Settlers settlers = Settlers.organizeBasedOn(world, data.numberOfSettlers());
		
		world = settlers.settleIndividualCellsOf(world);
		IoManager.askAboutAndPossiblyDisplaySettledWorld(world);
		
		//VonNeumannNeighborhood vonNeumannNeighborhood = new VonNeumannNeighborhood(data.rangeOfFarmingPattern());
		EuclidNeighborhood mooreNeighborhood = new EuclidNeighborhood(data.rangeOfFarmingPattern());
		IoManager.askAboutAndPossiblyDisplayFarmingPattern(mooreNeighborhood);
		
		world = settlers.farm(world, mooreNeighborhood);
		IoManager.askAboutAndPossiblyDisplaySettledAndFarmedWorld(world);
		
		System.out.println(
			"Neighbors in von-Neumann Neighborhoods / number of cells settled or farmed: " +
			world.numberOfCellsSettledOrFarmed());

	}

}
