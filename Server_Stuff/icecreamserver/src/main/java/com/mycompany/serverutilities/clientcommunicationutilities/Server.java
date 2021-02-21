
// Allows main to find class Server.
package com.mycompany.serverutilities.clientcommunicationutilities;

// Import classes.
import com.sun.net.httpserver.HttpServer;

/**
 * Defines class Server whose instances represent servers that listen for HTTP
 * messages from clients and provide responses to clients.
 * @version 0.0
 * @author Tom Lever
 */
public class Server {
    
    private final HttpServer httpServer;
    private MessageInterface[] messageInterfaces;
    private boolean setMessageInterfacesHasAlreadyBeenCalled;
    private boolean serverIsUnableToListenForMessages;
    
    /**
     * Defines constructor Server which sets this.httpServer as httpServerToUse
     * and sets setMessagesInterfacesHasAlreadyBeenCalled to false and
     * serverIsUnableToListenForMessages to true.
     * @param httpServerToUse
     */
    public Server(HttpServer httpServerToUse) {
        
        this.httpServer = httpServerToUse;
        setMessageInterfacesHasAlreadyBeenCalled = false;
        serverIsUnableToListenForMessages = true;
    }
    
    /**
     * Defines method setMessageInterfaces, which sets the array of message
     * interfaces of this server.
     * @version 0.0
     * @author Tom Lever
     * @param messageInterfacesToUse
     */
    void setMessageInterfaces(MessageInterface[] messageInterfacesToUse)
        throws SetMessageInterfacesException {
    // Without public, this method is "package-private"
    // to com.mycompany.serverutilities.clientcommunicationutilities.
        
        if (setMessageInterfacesHasAlreadyBeenCalled) {
            throw new SetMessageInterfacesException(
                "Message interfaces have already been set in Server.");
        }
        
        this.messageInterfaces = messageInterfacesToUse;
        
        for (MessageInterface messageInterface : this.messageInterfaces) {
            httpServer.createContext(
                messageInterface.getEndpoint(),
                messageInterface.getMessageHandler());
        }
        
        setMessageInterfacesHasAlreadyBeenCalled = true;
        serverIsUnableToListenForMessages = false;
    }
    
    /**
     * Defines method startListeningForMessages, which starts this.httpServer
     * listening for messages.
     * @version 0.0
     * @author Tom Lever
     * @throws StartServerListeningForMessagesException
     */
    public void startListeningForMessages()
        throws StartServerListeningForMessagesException {
        
        if (serverIsUnableToListenForMessages) {
            throw new StartServerListeningForMessagesException(
                "Server is unable to listen for messages: message interfaces " +
                "have not been set.");
        }
        
        this.httpServer.start();
    }
}