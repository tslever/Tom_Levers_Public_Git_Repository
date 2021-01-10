
// Allows executable to find class Main.
package com.mycompany.serverutilities.clientcommunicationutilities;

// Import classes.
import com.mycompany.serverutilities.productutilities.Products;
import com.sun.net.httpserver.HttpServer;
import java.util.logging.Level;
import java.util.logging.LogRecord;
import java.util.logging.Logger;

/**
 * Defines class Server whose instances represent servers that listen for HTTP
 * messages from clients and provide responses to clients.
 * @version 0.0
 * @author Tom Lever
 */
public class Server {
    
    private final static Logger logger =
        Logger.getLogger(Server.class.getName());
    
    private final HttpServer httpServer;
    private MessageInterface[] messageInterfaces;
    private boolean setMessageInterfacesHasAlreadyBeenCalled;
    private boolean serverIsUnableToListenForMessages;
    
    /**
     * Defines constructor Server which sets attributes of this server with
     * inputs.
     * @param httpServerToUse
     */
    public Server(HttpServer httpServerToUse) {
        logger.log(new LogRecord(Level.INFO,
            "Server constructor: Started."));
        
        this.httpServer = httpServerToUse;
        setMessageInterfacesHasAlreadyBeenCalled = false;
        serverIsUnableToListenForMessages = true;
        logger.log(new LogRecord(Level.INFO,
            "Server constructor: Set httpServer as httpServerToUse and noted " +
            "that setMessageInterfaces has not been called and that server " +
            "is unable to listen for messages."));
    }
    
    /**
     * Defines method setMessageInterfaces, which allows a calling method to
     * set the array of message interfaces of this server.
     * @version 0.0
     * @author Tom Lever
     * @param messageInterfacesToUse
     * @throws Exception
     */
    public void setMessageInterfaces(
            MessageInterface[] messageInterfacesToUse) throws Exception {
        logger.log(new LogRecord(Level.INFO,
            "Server.setMessageInterfaces: Started."));
        
        if (setMessageInterfacesHasAlreadyBeenCalled) {
            throw new Exception(
                "Server.setMessageInterfaces: Message interfaces have " +
                "already been set in Server.");
            // TODO: Throw a SetMessageInterfacesHasAlreadyBeenCalledException
            // instead of an Exception.
        }
        
        this.messageInterfaces = messageInterfacesToUse;
        logger.log(new LogRecord(Level.INFO,
            "Server.setMessageInterfaces: Stored inputted array of message " +
            "interfaces as this server's array of message interfaces."));
        
        for (MessageInterface messageInterface : this.messageInterfaces) {
            httpServer.createContext(
                messageInterface.getEndpoint(),
                messageInterface.getMessageHandler());
        }
        logger.log(new LogRecord(Level.INFO,
            "Server.setMessageInterfaces: Created a context to associate " +
            "a specific message handler with each endpoint for HTTP " +
            "messages."));
        
        setMessageInterfacesHasAlreadyBeenCalled = true;
        serverIsUnableToListenForMessages = false;
        logger.log(new LogRecord(Level.INFO,
            "Server.setMessageInterfaces: Noted that setMessageInterfaces " +
            "has been called and that server is able to listen for messages."));
    }
    
    /**
     * Defines method startListeningForMessages, which allows this server to
     * start listening for messages.
     * @version 0.0
     * @author Tom Lever
     * @throws Exception
     */
    public void startListeningForMessages() throws Exception {
        logger.log(new LogRecord(Level.INFO,
            "Server.startListeningForMessages: Started."));
        
        if (serverIsUnableToListenForMessages) {
            throw new Exception(
                "Server.starListeningForMessages: Server is unable to listen " +
                "for messages: message interfaces have not been created.");
            // TODO: Throw a ServerIsUnableToListenForMessagesException
            // instead of an Exception.
        }
        
        this.httpServer.start();
        logger.log(new LogRecord(Level.INFO,
            "Server.startListeningForMessages: Started server listening for " +
            "HTTP messages from clients."));
    }
    
    /**
     * Defines method send to send products corresponding to a message from an
     * ice cream client back to the ice cream client.
     * @param productsToSend
     */
    public void send(Products productsToSend) { }
}