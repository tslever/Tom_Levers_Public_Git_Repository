package lattice_utilities;


import settler_utilities.Location;


public abstract class FarmingPattern extends Lattice {

	
	protected int range;
	
	
	public FarmingPattern(int rangeToUse) {
		
		super(1+2*rangeToUse, 1+2*rangeToUse);
		
		this.range = rangeToUse;
		
	}
	
	
	protected abstract void fillCells();
	
	
	public abstract World applyTo(World world, Location location);
	
	
}
