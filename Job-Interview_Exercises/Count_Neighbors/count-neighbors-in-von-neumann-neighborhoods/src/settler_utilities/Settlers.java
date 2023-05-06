package settler_utilities;


import java.util.Random;
import lattice_utilities.FarmingPattern;
import lattice_utilities.World;


public class Settlers {

	private static Settlers settlers;
	private int numberOfSettlers;
	private Settler[] arrayOfSettlers;
	Random random = new Random();

	private Settlers(World world, int numberOfSettlersToUse) {

		this.numberOfSettlers = numberOfSettlersToUse;

		this.arrayOfSettlers = new Settler[this.numberOfSettlers];

		fillArrayOfSettlersBasedOn(world);

	}

	public static Settlers organizeBasedOn(World world, int numberOfSettlers) {

		if (settlers == null)
			settlers = new Settlers(world, numberOfSettlers);

		return settlers;

	}

	private int getRandomWholeNumberLessThan(int exclusiveMax) {

		return random.nextInt(exclusiveMax);
		// Abbreviated from random.nextInt(exclusiveMax - (inclusiveMin=0)) +
		// (inclusiveMin=0).

	}

	private void fillArrayOfSettlersBasedOn(World world) {

		int indexForNextSettler = 0;

		Location prospectiveLocation;
		while (indexForNextSettler < this.numberOfSettlers) {

			prospectiveLocation = new Location(getRandomWholeNumberLessThan(world.height()),
					getRandomWholeNumberLessThan(world.width()));

			if (aSettlerIsNotAlreadyOccupying(prospectiveLocation, indexForNextSettler)) {

				this.arrayOfSettlers[indexForNextSettler] = new Settler(prospectiveLocation);

				indexForNextSettler++;

			}

		}

	}

	private boolean aSettlerIsNotAlreadyOccupying(Location prospectiveLocation, int indexForNextSettler) {

		Settler establishedSettler;

		for (int i = 0; i < indexForNextSettler; i++) {

			establishedSettler = this.arrayOfSettlers[i];

			if ((prospectiveLocation.row() == establishedSettler.location().row())
					&& (prospectiveLocation.column() == establishedSettler.location().column())) {

				return false;

			}

		}

		return true;

	}

	public World settleIndividualCellsOf(World world) {

		for (Settler settler : this.arrayOfSettlers) {

			world.setCellIdentifiedBy(settler.location().row(), settler.location().column());
			world.incrementNumberOfCellsSettledOrFarmed();

		}

		return world;

	}
	
	
	public World farm(World world, FarmingPattern farmingPattern) {
		
		for (Settler settler : this.arrayOfSettlers) {
			
			world = settler.farm(world, farmingPattern);
			
		}
		
		return world;
		
	}
	

}
