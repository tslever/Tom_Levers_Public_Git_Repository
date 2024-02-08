package Com.TSL.Settles_Of_Catan;

public class Road {
	
	public static enum Type {
		NO_ROAD,
		ROAD
	}
	
	private int xCoordinate;
	private int yCoordinate;
	private Type type;
	private int affiliation;

	public Road(int xCoordinateToUse, int yCoordinateToUse, Type typeToUse, int affiliationToUse) {
		xCoordinate = xCoordinateToUse;
		yCoordinate = yCoordinateToUse;
		type = typeToUse;
		affiliation = affiliationToUse;
	}
	
	public int xCoordinate() {
		return xCoordinate;
	}
	
	public int yCoordinate() {
		return yCoordinate;
	}
	
	public Type type() {
		return type;
	}
	
	public int affiliation() {
		return affiliation;
	}
}
