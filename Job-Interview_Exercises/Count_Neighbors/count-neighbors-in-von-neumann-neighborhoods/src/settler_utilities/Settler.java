package settler_utilities;


import lattice_utilities.FarmingPattern;
import lattice_utilities.World;


public class Settler {
	
	
	private Location location;
	
	
	public Settler(Location locationToUse) {
		
		this.location = locationToUse;
		
	}
	
	
	public Location location() {
		
		return this.location;
		
	}
	
	
	public World farm(World world, FarmingPattern farmingPattern) {
		
		world = farmingPattern.applyTo(world, this.location);
		
		return world;
		
	}
	

}
