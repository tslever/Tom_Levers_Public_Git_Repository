package settler_utilities;

public class Location {

	
	private int row;
	private int column;
	
	
	public Location(int rowToUse, int columnToUse) {
		
		this.row = rowToUse;
		this.column = columnToUse;
		
	}
	
	
	public int row() {
		
		return this.row;
	
	}
	
	
	public int column() {
		
		return this.column;
		
	}
	
	
}
