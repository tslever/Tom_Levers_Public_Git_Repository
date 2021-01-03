
/*
 * Distinguishes this design class from Server in package
 * com.mycompany.serverutilities.
 */
package com.mycompany.designclasses;

// Import classes.
import com.sun.net.httpserver.HttpServer;

/**
 * Defines class Server whose instances represent servers that listen for HTTP
 * messages from clients and provide responses to clients.
 * @version 0.0
 * @author Tom Lever
 */
public class Server {
    
    // Commented out to allow building.
    //private final HttpServer httpServer;
    private MessageInterface[] messageInterfaces;
    private boolean setMessageInterfacesHasAlreadyBeenCalled;
    private boolean serverIsUnableToListenForMessages;
    
    /**
     * Defines constructor Server which sets attributes of this server with
     * inputs.
     * @param httpServerToUse
     */
    public Server(HttpServer httpServerToUse) { }
    
    /**
     * Defines method setMessageInterfaces, which allows a calling method to
     * set the array of message interfaces of this server.
     * @version 0.0
     * @author Tom Lever
     * @param messageInterfacesToUse
     * @throws Exception
     */
    public void setMessageInterfaces(
            MessageInterface[] messageInterfacesToUse) throws Exception { }
    
    /**
     * Defines method startListeningForMessages, which allows this server to
     * start listening for messages.
     * @version 0.0
     * @author Tom Lever
     * @throws Exception
     */
    public void startListeningForMessages() throws Exception { }
    
    /**
     * Defines method send to send products corresponding to a message from an
     * ice cream client back to the ice cream client.
     * @param productsToSend
     */
    public void send(Products productsToSend) { }
}