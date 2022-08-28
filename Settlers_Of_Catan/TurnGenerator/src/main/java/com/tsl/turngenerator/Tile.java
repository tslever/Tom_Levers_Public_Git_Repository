package com.tsl.turngenerator;

public class Tile {

	private int xCoordinate;
	private int yCoordinate;
	private Resource resource;
	private NumberToken numberToken;
	private boolean hasRobber;
	private Community[] communities;
	
	public Tile(int xCoordinateToUse, int yCoordinateToUse, Resource resourceToUse, NumberToken numberTokenToUse, boolean robberIndication) {
		xCoordinate = xCoordinateToUse;
		yCoordinate = yCoordinateToUse;
		resource = resourceToUse;
		numberToken = numberTokenToUse;
		hasRobber = robberIndication;
	}
	
	public int xCoordinate() {
		return xCoordinate;
	}
	
	public int yCoordinate() {
		return yCoordinate;
	}
	
	public Resource resource() {
		return resource;
	}
	
	public NumberToken numberToken() {
		return numberToken;
	}
	
	public boolean hasRobber() {
		return hasRobber;
	}
	
	public void setCommunities(Community... communitiesToUse) {
		communities = communitiesToUse;
	}
}
