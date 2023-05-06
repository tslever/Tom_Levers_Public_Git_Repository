package neighbor_counting_utilities;

public class Data {
	
	private int heightOfWorld;
	private int widthOfWorld;
	private int numberOfSettlers;
	private int rangeOfFarmingPattern;
	
	public Data(int[] heightWidthNumberOfSettlersAndRange) {
		
		this.heightOfWorld = heightWidthNumberOfSettlersAndRange[0];
		this.widthOfWorld = heightWidthNumberOfSettlersAndRange[1];
		this.numberOfSettlers = heightWidthNumberOfSettlersAndRange[2];
		this.rangeOfFarmingPattern = heightWidthNumberOfSettlersAndRange[3];
		
	}
	
	public int heightOfWorld() {
		return this.heightOfWorld;
	}
	
	public int widthOfWorld() {
		return this.widthOfWorld;
	}
	
	public int numberOfSettlers() {
		return this.numberOfSettlers;
	}
	
	public int rangeOfFarmingPattern() {
		return this.rangeOfFarmingPattern;
	}

}
