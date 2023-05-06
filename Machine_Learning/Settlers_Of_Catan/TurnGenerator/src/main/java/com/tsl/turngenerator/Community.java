package com.tsl.turngenerator;

public class Community {
	
	public static enum Type {
		NO_COMMUNITY,
		SETTLEMENT,
		CITY
	}
	
	private int xCoordinate;
	private int yCoordinate;
	private Type type;
	private int affiliation;
	
	public Community(int xCoordinateToUse, int yCoordinateToUse, Type typeToUse, int affiliationToUse) {
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
