package Com.TSL.Settles_Of_Catan;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

public class Turn {

	private final int WIDTH  = 21;
	private final int HEIGHT = 21;
	private final int DEPTH  =  3;
	private final int INDEX_OF_RED_CHANNEL = 0;
	private final int INDEX_OF_GREEN_CHANNEL = 1;
	private final int INDEX_OF_BLUE_CHANNEL = 2;
	private final float MAXIMUM_INTENSITY = 255.0f;
	
	private int[][][] tensor = new int[HEIGHT][WIDTH][DEPTH];
	
	public Turn(Board board, Bank bank, Player[] players) throws IOException {
		
		for (Tile tile : board.tiles()) {
			tensor[tile.xCoordinate() + 10][tile.yCoordinate() + 10][ INDEX_OF_BLUE_CHANNEL] = Math.round((tile.hasRobber() ? 2.0f : 0.0f) * MAXIMUM_INTENSITY / 2.0f);
			tensor[tile.xCoordinate() + 10][tile.yCoordinate() + 10][INDEX_OF_GREEN_CHANNEL] = Math.round(((float) (NumberToken.values.indexOf(tile.numberToken().value()) + 1)) * MAXIMUM_INTENSITY / ((float) NumberToken.values.size()));
			tensor[tile.xCoordinate() + 10][tile.yCoordinate() + 10][  INDEX_OF_RED_CHANNEL] = Math.round(((float) (tile.resource().ordinal() + 1)) * MAXIMUM_INTENSITY / ((float) Resource.values().length));
		}
		for (Community community : board.communities()) {
			tensor[community.xCoordinate() + 10][community.yCoordinate() + 10][INDEX_OF_GREEN_CHANNEL] = Math.round(((float) (community.affiliation() + 1)) * MAXIMUM_INTENSITY / ((float) players.length));
			tensor[community.xCoordinate() + 10][community.yCoordinate() + 10][  INDEX_OF_RED_CHANNEL] = Math.round(((float) (community.type().ordinal() + 1)) * MAXIMUM_INTENSITY / ((float) Community.Type.values().length));
		}
		for (Road road : board.roads()) {
			tensor[road.xCoordinate() + 10][road.yCoordinate() + 10][INDEX_OF_GREEN_CHANNEL] = Math.round(((float) (road.affiliation() + 1)) * MAXIMUM_INTENSITY / ((float) players.length));
			tensor[road.xCoordinate() + 10][road.yCoordinate() + 10][  INDEX_OF_RED_CHANNEL] = Math.round(((float) (road.type().ordinal() + 1)) * MAXIMUM_INTENSITY / ((float) Road.Type.values().length));
		}
		
		tensor[Resource. BRICK.ordinal()][HEIGHT - 1][INDEX_OF_RED_CHANNEL] = 153; // Math.round(((float) bank.      brickCards()) / ((float) Bank.INITIAL_NUMBER_OF_CARDS_OF_ONE_RESOURCE) * MAXIMUM_INTENSITY);
		tensor[Resource.LUMBER.ordinal()][HEIGHT - 1][INDEX_OF_RED_CHANNEL] = 179; // Math.round(((float) bank.     lumberCards()) / ((float) Bank.INITIAL_NUMBER_OF_CARDS_OF_ONE_RESOURCE) * MAXIMUM_INTENSITY);
		tensor[Resource.   ORE.ordinal()][HEIGHT - 1][INDEX_OF_RED_CHANNEL] = 204; // Math.round(((float) bank.        oreCards()) / ((float) Bank.INITIAL_NUMBER_OF_CARDS_OF_ONE_RESOURCE) * MAXIMUM_INTENSITY);
		tensor[Resource. GRAIN.ordinal()][HEIGHT - 1][INDEX_OF_RED_CHANNEL] = 230; // Math.round(((float) bank.      grainCards()) / ((float) Bank.INITIAL_NUMBER_OF_CARDS_OF_ONE_RESOURCE) * MAXIMUM_INTENSITY);
		tensor[Resource.  WOOL.ordinal()][HEIGHT - 1][INDEX_OF_RED_CHANNEL] = 255; // Math.round(((float) bank.       woolCards()) / ((float) Bank.INITIAL_NUMBER_OF_CARDS_OF_ONE_RESOURCE) * MAXIMUM_INTENSITY);
		
		tensor[WIDTH - 1 - 0][0][INDEX_OF_RED_CHANNEL] = 128; // Math.round(((float) bank.developmentCards()) / ((float) Bank.INITIAL_NUMBER_OF_DEVELOPMENT_CARDS) * MAXIMUM_INTENSITY);
		tensor[WIDTH - 1 - 1][0][INDEX_OF_RED_CHANNEL] = 153; // Math.round(                             bank.      probabilityThatADevelopmentCardIsAKnightCard() * MAXIMUM_INTENSITY);
		tensor[WIDTH - 1 - 2][0][INDEX_OF_RED_CHANNEL] = 179; // Math.round(                             bank.probabilityThatADevelopmentCardIsAVictoryPointCard() * MAXIMUM_INTENSITY);
		tensor[WIDTH - 1 - 3][0][INDEX_OF_RED_CHANNEL] = 204; // Math.round(                             bank.probabilityThatADevelopmentCardIsARoadBuildingCard() * MAXIMUM_INTENSITY);
		tensor[WIDTH - 1 - 4][0][INDEX_OF_RED_CHANNEL] = 230; // Math.round(                             bank.probabilityThatADevelopmentCardIsAYearOfPlentyCard() * MAXIMUM_INTENSITY);
		tensor[WIDTH - 1 - 5][0][INDEX_OF_RED_CHANNEL] = 255; // Math.round(                             bank.    probabilityThatADevelopmentCardIsAMonopolyCard() * MAXIMUM_INTENSITY);
		
		Player[] orderedPlayers = order(players);
		Player activePlayer = orderedPlayers[0];
		tensor[                        0][HEIGHT - 2][INDEX_OF_RED_CHANNEL] = 128; // Math.round(((float) activePlayer.     victoryPoints()) / ((float) Bank.INITIAL_NUMBER_OF_CARDS_OF_ONE_RESOURCE) * MAXIMUM_INTENSITY);
		tensor[Resource. BRICK.ordinal()][HEIGHT - 2][INDEX_OF_RED_CHANNEL] = 153; // Math.round(((float) activePlayer.hand(). brickCards()) / ((float) Bank.INITIAL_NUMBER_OF_CARDS_OF_ONE_RESOURCE) * MAXIMUM_INTENSITY);
		tensor[Resource.LUMBER.ordinal()][HEIGHT - 2][INDEX_OF_RED_CHANNEL] = 179; // Math.round(((float) activePlayer.hand().lumberCards()) / ((float) Bank.INITIAL_NUMBER_OF_CARDS_OF_ONE_RESOURCE) * MAXIMUM_INTENSITY);
		tensor[Resource.   ORE.ordinal()][HEIGHT - 2][INDEX_OF_RED_CHANNEL] = 204; // Math.round(((float) activePlayer.hand().   oreCards()) / ((float) Bank.INITIAL_NUMBER_OF_CARDS_OF_ONE_RESOURCE) * MAXIMUM_INTENSITY);
		tensor[Resource. GRAIN.ordinal()][HEIGHT - 2][INDEX_OF_RED_CHANNEL] = 230; // Math.round(((float) activePlayer.hand(). grainCards()) / ((float) Bank.INITIAL_NUMBER_OF_CARDS_OF_ONE_RESOURCE) * MAXIMUM_INTENSITY);
		tensor[Resource.  WOOL.ordinal()][HEIGHT - 2][INDEX_OF_RED_CHANNEL] = 255; // Math.round(((float) activePlayer.hand().  woolCards()) / ((float) Bank.INITIAL_NUMBER_OF_CARDS_OF_ONE_RESOURCE) * MAXIMUM_INTENSITY);
		
		tensor[WIDTH - 1 - 0][1][INDEX_OF_RED_CHANNEL] = 128; // Math.round(((float) activePlayer.hand(). developmentCards()) / ((float) Bank.INITIAL_NUMBER_OF_DEVELOPMENT_CARDS) * MAXIMUM_INTENSITY);
		tensor[WIDTH - 1 - 1][1][INDEX_OF_RED_CHANNEL] = 153; // Math.round(((float) activePlayer.hand().      knightCards()) / ((float) Bank.INITIAL_NUMBER_OF_DEVELOPMENT_CARDS) * MAXIMUM_INTENSITY);
		tensor[WIDTH - 1 - 2][1][INDEX_OF_RED_CHANNEL] = 179; // Math.round(((float) activePlayer.hand().victoryPointCards()) / ((float) Bank.INITIAL_NUMBER_OF_DEVELOPMENT_CARDS) * MAXIMUM_INTENSITY);
		tensor[WIDTH - 1 - 3][1][INDEX_OF_RED_CHANNEL] = 204; // Math.round(((float) activePlayer.hand().roadBuildingCards()) / ((float) Bank.INITIAL_NUMBER_OF_DEVELOPMENT_CARDS) * MAXIMUM_INTENSITY);
		tensor[WIDTH - 1 - 4][1][INDEX_OF_RED_CHANNEL] = 230; // Math.round(((float) activePlayer.hand().yearOfPlentyCards()) / ((float) Bank.INITIAL_NUMBER_OF_DEVELOPMENT_CARDS) * MAXIMUM_INTENSITY);
		tensor[WIDTH - 1 - 5][1][INDEX_OF_RED_CHANNEL] = 255; // Math.round(((float) activePlayer.hand().    monopolyCards()) / ((float) Bank.INITIAL_NUMBER_OF_DEVELOPMENT_CARDS) * MAXIMUM_INTENSITY);
		
		for (int i = 1; i < orderedPlayers.length; i++) {
			Player player = orderedPlayers[i];
			
			tensor[                        0][HEIGHT - 2 - i][INDEX_OF_RED_CHANNEL] = 128; // Math.round(((float) player.     victoryPoints()) / ((float) Bank.INITIAL_NUMBER_OF_CARDS_OF_ONE_RESOURCE) * MAXIMUM_INTENSITY);
			tensor[Resource. BRICK.ordinal()][HEIGHT - 2 - i][INDEX_OF_RED_CHANNEL] = 153; // Math.round(((float) player.hand(). brickCards()) / ((float) Bank.INITIAL_NUMBER_OF_CARDS_OF_ONE_RESOURCE) * MAXIMUM_INTENSITY);
			tensor[Resource.LUMBER.ordinal()][HEIGHT - 2 - i][INDEX_OF_RED_CHANNEL] = 179; // Math.round(((float) player.hand().lumberCards()) / ((float) Bank.INITIAL_NUMBER_OF_CARDS_OF_ONE_RESOURCE) * MAXIMUM_INTENSITY);
			tensor[Resource.   ORE.ordinal()][HEIGHT - 2 - i][INDEX_OF_RED_CHANNEL] = 204; // Math.round(((float) player.hand().   oreCards()) / ((float) Bank.INITIAL_NUMBER_OF_CARDS_OF_ONE_RESOURCE) * MAXIMUM_INTENSITY);
			tensor[Resource. GRAIN.ordinal()][HEIGHT - 2 - i][INDEX_OF_RED_CHANNEL] = 230; // Math.round(((float) player.hand(). grainCards()) / ((float) Bank.INITIAL_NUMBER_OF_CARDS_OF_ONE_RESOURCE) * MAXIMUM_INTENSITY);
			tensor[Resource.  WOOL.ordinal()][HEIGHT - 2 - i][INDEX_OF_RED_CHANNEL] = 255; // Math.round(((float) player.hand().  woolCards()) / ((float) Bank.INITIAL_NUMBER_OF_CARDS_OF_ONE_RESOURCE) * MAXIMUM_INTENSITY);
			
			tensor[WIDTH - 1 - 0][1 + player.id()][INDEX_OF_RED_CHANNEL] = 128; // Math.round(((float) player.hand().developmentCards()) / ((float) Bank.INITIAL_NUMBER_OF_DEVELOPMENT_CARDS) * MAXIMUM_INTENSITY);
			tensor[WIDTH - 1 - 1][1 + player.id()][INDEX_OF_RED_CHANNEL] = 153; // Math.round(                             player.hand().      probabilityThatADevelopmentCardIsAKnightCard() * MAXIMUM_INTENSITY);
			tensor[WIDTH - 1 - 2][1 + player.id()][INDEX_OF_RED_CHANNEL] = 179; // Math.round(                             player.hand().probabilityThatADevelopmentCardIsAVictoryPointCard() * MAXIMUM_INTENSITY);
			tensor[WIDTH - 1 - 3][1 + player.id()][INDEX_OF_RED_CHANNEL] = 204; // Math.round(                             player.hand().probabilityThatADevelopmentCardIsARoadBuildingCard() * MAXIMUM_INTENSITY);
			tensor[WIDTH - 1 - 4][1 + player.id()][INDEX_OF_RED_CHANNEL] = 230; // Math.round(                             player.hand().probabilityThatADevelopmentCardIsAYearOfPlentyCard() * MAXIMUM_INTENSITY);
			tensor[WIDTH - 1 - 5][1 + player.id()][INDEX_OF_RED_CHANNEL] = 255; // Math.round(                             player.hand().    probabilityThatADevelopmentCardIsAMonopolyCard() * MAXIMUM_INTENSITY);
		}
	}
	
	public void enterVictoryPointsAndNumbersOfResources(Player player) {

	}
	
	public int getIdOfActivePlayer(Player[] players) throws IOException {
		for (Player player : players) {
			if (player.isActive()) {
				return player.id();
			}
		}
		throw new IOException("No player is active.");
	}
	
	public Player[] order(Player[] players) throws IOException {
		Player[] orderedPlayers = new Player[players.length];
		int indexOfCurrentPlayer = getIdOfActivePlayer(players);
		for (int i = 0; i < players.length; i++) {
			if (indexOfCurrentPlayer == players.length) {
				indexOfCurrentPlayer = 0;
			}
			orderedPlayers[i] = players[indexOfCurrentPlayer];
			indexOfCurrentPlayer++;
		}
		return orderedPlayers;
	}
	
	public void write(String pathRelativeToProjectWithoutFirstSlash) throws IOException {
		File file = new File(pathRelativeToProjectWithoutFirstSlash);
		BufferedImage bufferedImage = new BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_3BYTE_BGR);
		for (int i = 0; i < WIDTH; i++) {
			for (int j = 0; j < HEIGHT; j++) {
				bufferedImage.setRGB(i, j, (new Color(tensor[i][j][INDEX_OF_RED_CHANNEL], tensor[i][j][INDEX_OF_GREEN_CHANNEL], tensor[i][j][INDEX_OF_BLUE_CHANNEL])).getRGB());
			}
		}
		ImageIO.write(bufferedImage, "PNG", file);
	}
}
