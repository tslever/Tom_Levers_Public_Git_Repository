package Com.TSL.Settles_Of_Catan;

public class Player {

	private int id;
	private int victoryPoints;
	private Hand hand;
	private boolean isActive;
	
	public Player(int idToUse) {
		id = idToUse;
		victoryPoints = 0;
		hand = new Hand();
	}
	
	public Hand hand() {
		return hand;
	}
	
	public int victoryPoints() {
		return victoryPoints;
	}
	
	public int id() {
		return id;
	}
	
	public void activate() {
		isActive = true;
	}
	
	public void deactivate() {
		isActive = false;
	}
	
	public boolean isActive() {
		return isActive;
	}
}
