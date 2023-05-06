package lattice_utilities;


public class World extends Lattice {

	
	private boolean wrapsVertically;
	private boolean wrapsHorizontally;
	private int numberOfCellsSettledOrFarmed;
	
	
	public World(int heightToUse, int widthToUse) {
		
		super(heightToUse, widthToUse);
		
		this.wrapsVertically = (heightToUse < 3) ? true : false;
		
		this.wrapsHorizontally = (widthToUse < 3) ? true : false;
		
		this.numberOfCellsSettledOrFarmed = 0;
		
	}
	
	
	public boolean wrapsVertically() {
		
		return this.wrapsVertically;
		
	}
	
	
	public boolean wrapsHorizontally() {
		
		return this.wrapsHorizontally;
		
	}
	
	
	public void incrementNumberOfCellsSettledOrFarmed() {
		
		this.numberOfCellsSettledOrFarmed = this.numberOfCellsSettledOrFarmed + 1;
		
	}
	
	
	public int numberOfCellsSettledOrFarmed() {
		
		return this.numberOfCellsSettledOrFarmed;
		
	}
	
	
}