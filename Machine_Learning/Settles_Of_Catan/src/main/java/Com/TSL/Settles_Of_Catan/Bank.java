package Com.TSL.Settles_Of_Catan;

public class Bank {
	
	public static final int INITIAL_NUMBER_OF_CARDS_OF_ONE_RESOURCE = 19;
	public static final int     INITIAL_NUMBER_OF_DEVELOPMENT_CARDS = 25;
	public static final int          INITIAL_NUMBER_OF_KNIGHT_CARDS = 14;
	public static final int    INITIAL_NUMBER_OF_VICTORY_POINT_CARDS = 5;
	public static final int    INITIAL_NUMBER_OF_ROAD_BUILDING_CARDS = 2;
	public static final int   INITIAL_NUMBER_OF_YEAR_OF_PLENTY_CARDS = 2;
	public static final int         INITIAL_NUMBER_OF_MONOPOLY_CARDS = 2;

	private int brickCards;
	private int lumberCards;
	private int oreCards;
	private int grainCards;
	private int woolCards;
	private int knightCards;
	private int victoryPointCards;
	private int roadBuildingCards;
	private int yearOfPlentyCards;
	private int monopolyCards;
	private float probabilityThatADevelopmentCardIsAKnightCard;
	private float probabilityThatADevelopmentCardIsAVictoryPointCard;
	private float probabilityThatADevelopmentCardIsARoadBuildingCard;
	private float probabilityThatADevelopmentCardIsAYearOfPlentyCard;
	private float probabilityThatADevelopmentCardIsAMonopolyCard;
	
	public Bank() {
		brickCards = 19;
		lumberCards = 19;
		oreCards = 19;
		grainCards = 19;
		woolCards = 19;
		knightCards = INITIAL_NUMBER_OF_KNIGHT_CARDS;
		victoryPointCards = INITIAL_NUMBER_OF_VICTORY_POINT_CARDS;
		roadBuildingCards = INITIAL_NUMBER_OF_ROAD_BUILDING_CARDS;
		yearOfPlentyCards = INITIAL_NUMBER_OF_YEAR_OF_PLENTY_CARDS;
		monopolyCards = INITIAL_NUMBER_OF_MONOPOLY_CARDS;
		probabilityThatADevelopmentCardIsAKnightCard       = ((float)         INITIAL_NUMBER_OF_KNIGHT_CARDS) / ((float) INITIAL_NUMBER_OF_DEVELOPMENT_CARDS);
		probabilityThatADevelopmentCardIsAVictoryPointCard = ((float)  INITIAL_NUMBER_OF_VICTORY_POINT_CARDS) / ((float) INITIAL_NUMBER_OF_DEVELOPMENT_CARDS);
		probabilityThatADevelopmentCardIsARoadBuildingCard = ((float)  INITIAL_NUMBER_OF_ROAD_BUILDING_CARDS) / ((float) INITIAL_NUMBER_OF_DEVELOPMENT_CARDS);
		probabilityThatADevelopmentCardIsAYearOfPlentyCard = ((float) INITIAL_NUMBER_OF_YEAR_OF_PLENTY_CARDS) / ((float) INITIAL_NUMBER_OF_DEVELOPMENT_CARDS);
		probabilityThatADevelopmentCardIsAMonopolyCard     = ((float)       INITIAL_NUMBER_OF_MONOPOLY_CARDS) / ((float) INITIAL_NUMBER_OF_DEVELOPMENT_CARDS);
	}
	
	public int brickCards() {
		return brickCards;
	}
	
	public int lumberCards() {
		return lumberCards;
	}
	
	public int oreCards() {
		return oreCards;
	}
	
	public int grainCards() {
		return grainCards;
	}
	
	public int woolCards() {
		return woolCards;
	}
	
	public int knightCards() {
		return knightCards;
	}
	
	public int victoryPointCards() {
		return victoryPointCards;
	}
	
	public int roadBuildingCards() {
		return roadBuildingCards;
	}
	
	public int yearOfPlentyCards() {
		return yearOfPlentyCards;
	}
	
	public int monopolyCards() {
		return monopolyCards;
	}
	
	public int developmentCards() {
		return knightCards + victoryPointCards + roadBuildingCards + yearOfPlentyCards + monopolyCards;
	}
	
	public float probabilityThatADevelopmentCardIsAKnightCard() {
		return probabilityThatADevelopmentCardIsAKnightCard;
	}
	
	public float probabilityThatADevelopmentCardIsAVictoryPointCard() {
		return probabilityThatADevelopmentCardIsAVictoryPointCard;
	}
	
	public float probabilityThatADevelopmentCardIsARoadBuildingCard() {
		return probabilityThatADevelopmentCardIsARoadBuildingCard;
	}
	
	public float probabilityThatADevelopmentCardIsAYearOfPlentyCard() {
		return probabilityThatADevelopmentCardIsAYearOfPlentyCard;
	}
	
	public float probabilityThatADevelopmentCardIsAMonopolyCard() {
		return probabilityThatADevelopmentCardIsAMonopolyCard;
	}
}