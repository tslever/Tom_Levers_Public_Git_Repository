package lattice_utilities;


import settler_utilities.Location;


public class EuclidNeighborhood extends FarmingPattern {

	
	public EuclidNeighborhood(int range) {
		
		super(range);
		
		fillCells();
		
	}
	
	
	protected void fillCells() {
		
		
		int j;
	
		for (int i = 0; i < this.height; i++) {
			
			for (j = 0; j < this.width; j++) {
				
				if ( Math.sqrt( Math.pow(this.range-i,2) + Math.pow(this.range-j,2) ) <= this.range )
					setCellIdentifiedBy(i,j);
				
			}
			
		}
		
	}
	
	
	public World applyTo(World world, Location location) {
		
		if (world.wrapsVertically() && world.wrapsHorizontally())
			return byUsingMethodForWhenWorldWrapsVerticallyAndHorizontally(world, location);
		
		else if (world.wrapsVertically() && !world.wrapsHorizontally())
			return byUsingMethodForWhenWorldWrapsVerticallyButNotHorizontally(world, location);
		
		else if (!world.wrapsVertically() && world.wrapsHorizontally())
			return byUsingMethodForWhenWorldDoesNotWrapVerticallyButWrapsHorizontally(world, location);
		
		else return byUsingMethodForWhenWorldWrapsNeitherVerticallyNorHorizontally(world, location);
		
	}
	
	
	private World byUsingMethodForWhenWorldWrapsVerticallyAndHorizontally(World world, Location location) {
		
		int j;
		int row;
		int column;
		
		for (int i = location.row() - this.range; i < location.row() + this.range + 1; i++) {
			
			for (j = location.column() - this.range; j < location.column() + this.range; j++) {
				
				if ( Math.sqrt( Math.pow(location.row()-i, 2) + Math.pow(location.column()-j, 2) ) <= this.range ) {
				
					row = i;
					while (row < 0) row += world.height;
					while (row >= world.height) row -= world.height;
					
					column = j;
					while (column < 0) column += world.width;
					while (column >= world.width) column -= world.width;
					
					world = farmCellIfNotAlreadyFarmed(world, row, column);
					
				}
				
			}
			
		}
		
		return world;
		
	}
	
	
	private World byUsingMethodForWhenWorldWrapsVerticallyButNotHorizontally(World world, Location location) {
		
		int j;
		int row;
		
		for (int i = location.row() - this.range; i < location.row() + this.range + 1; i++) {
			
			for (j = Math.max(0, location.column() - this.range); j < Math.min(world.width(), location.column() + this.range + 1); j++) {
				
				if ( Math.sqrt( Math.pow(location.row()-i, 2) + Math.pow(location.column()-j, 2) ) <= this.range ) {
				
					row = i;
					while (row < 0) row += world.height;
					while (row >= world.height) row -= world.height;
					
					world = farmCellIfNotAlreadyFarmed(world, row, j);
				
				}
				
			}
			
		}
		
		return world;
		
	}
	
	
	private World byUsingMethodForWhenWorldDoesNotWrapVerticallyButWrapsHorizontally(World world, Location location) {
		
		int j;
		int column;
		
		for (int i = Math.max(0, location.row() - this.range); i < Math.min(world.height(), location.row() + this.range + 1); i++) {
			
			for (j = location.column() - this.range + Math.abs(i - location.row()); j < location.column() + this.range - Math.abs(i - location.row()) + 1; j++) {
				
				if ( Math.sqrt( Math.pow(location.row()-i, 2) + Math.pow(location.column()-j, 2) ) <= this.range ) {
				
					column = j;
					while (column < 0) column += world.width;
					while (column >= world.width) column -= world.width;
					
					world = farmCellIfNotAlreadyFarmed(world, i, column);
				
				}
				
			}
			
		}
		
		return world;
		
	}
	
	
	private World byUsingMethodForWhenWorldWrapsNeitherVerticallyNorHorizontally(World world, Location location) {
		
		int j;
		
		for (int i = Math.max(0, location.row() - this.range); i < Math.min(world.height(), location.row() + this.range + 1); i++) {
			
			for (j = Math.max(0, location.column() - this.range); j < Math.min(world.width(), location.column() + this.range + 1); j++) {
				
				if ( Math.sqrt( Math.pow(location.row()-i, 2) + Math.pow(location.column()-j, 2) ) <= this.range ) {
				
					world = farmCellIfNotAlreadyFarmed(world, i, j);
				
				}
				
			}
			
		}
		
		return world;
		
	}
	
	
	private World farmCellIfNotAlreadyFarmed(World world, int row, int column) {
		
		if (world.valueOfCellIdentifiedBy(row, column) == 0) {
			world.setCellIdentifiedBy(row, column);
			world.incrementNumberOfCellsSettledOrFarmed();
		}
		
		return world;
		
	}
	
	
}