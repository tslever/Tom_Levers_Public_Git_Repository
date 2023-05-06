package com.tsl.turngenerator;

public class Hand {

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
	
	public Hand() {
		brickCards = 0;
		lumberCards = 0;
		oreCards = 0;
		grainCards = 0;
		woolCards = 0;
		knightCards = 0;
		victoryPointCards = 0;
		roadBuildingCards = 0;
		yearOfPlentyCards = 0;
		monopolyCards = 0;
		probabilityThatADevelopmentCardIsAKnightCard       = ((float)         Bank.INITIAL_NUMBER_OF_KNIGHT_CARDS) / ((float) Bank.INITIAL_NUMBER_OF_DEVELOPMENT_CARDS);
		probabilityThatADevelopmentCardIsAVictoryPointCard = ((float)  Bank.INITIAL_NUMBER_OF_VICTORY_POINT_CARDS) / ((float) Bank.INITIAL_NUMBER_OF_DEVELOPMENT_CARDS);
		probabilityThatADevelopmentCardIsARoadBuildingCard = ((float)  Bank.INITIAL_NUMBER_OF_ROAD_BUILDING_CARDS) / ((float) Bank.INITIAL_NUMBER_OF_DEVELOPMENT_CARDS);
		probabilityThatADevelopmentCardIsAYearOfPlentyCard = ((float) Bank.INITIAL_NUMBER_OF_YEAR_OF_PLENTY_CARDS) / ((float) Bank.INITIAL_NUMBER_OF_DEVELOPMENT_CARDS);
		probabilityThatADevelopmentCardIsAMonopolyCard     = ((float)       Bank.INITIAL_NUMBER_OF_MONOPOLY_CARDS) / ((float) Bank.INITIAL_NUMBER_OF_DEVELOPMENT_CARDS);
		
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