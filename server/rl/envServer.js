const express = require('express');
const GameFlowWrapper = require('../../../forceteki/test/helpers/GameFlowWrapper.js');
const { UnitTestCardDataGetter } = require('../../../forceteki/server/utils/cardData/UnitTestCardDataGetter.js');
const { UndoMode } = require('../../../forceteki/server/game/core/snapshot/SnapshotManager.js');

const app = express();
app.use(express.json());

let gameFlowWrapper = null;
let gameStateBuilder = null;

// Mock Router
const mockRouter = {
    gameWon: () => {},
    playerLeft: () => {},
    handleError: (game, error) => { console.error(error); },
    handleGameEnd: () => {},
    handleUndoGameEnd: () => {}
};

app.post('/reset', async (req, res) => {
    try {
        const { GameStateBuilder } = require('../../../forceteki/test/helpers/GameStateBuilder.js');
        const gameStateBuilder = new GameStateBuilder();
        
        const testContext = {};
        const directory = '../../test/json';
        const { UnitTestCardDataGetter } = require('../../../forceteki/server/utils/cardData/UnitTestCardDataGetter.js');
        const cardDataGetter = new UnitTestCardDataGetter(directory);
        
        gameFlowWrapper = new GameFlowWrapper(
            cardDataGetter,
            mockRouter,
            { id: '111', username: 'player1', settings: { optionSettings: { autoSingleTarget: false } } },
            { id: '222', username: 'player2', settings: { optionSettings: { autoSingleTarget: false } } },
            UndoMode.Disabled
        );
        
        gameStateBuilder.attachTestInfoToObj(testContext, gameFlowWrapper, 'player1', 'player2');
        
        // This correctly sets up the decks using test conventions and calls gameFlowWrapper.startGameAsync
        await gameStateBuilder.setupGameStateAsync(testContext, req.body.options || {});
        
        res.json(getState());
    } catch (e) {
        console.error(e);
        res.status(500).json({ error: e.message });
    }
});

app.post('/step', (req, res) => {
    try {
        if (!gameFlowWrapper) throw new Error("Game not initialized. Call /reset first.");
        
        const { playerId, action, arg, uuid, method, promptText } = req.body;
        
        // Find player interaction wrapper
        const playerWrapper = gameFlowWrapper.allPlayers.find(p => p.id === playerId);
        
        if (action === 'clickPrompt' && promptText) {
            playerWrapper.clickPrompt(promptText);
        } else if (action === 'clickCard' && uuid) {
             gameFlowWrapper.game.cardClicked(playerId, uuid);
             gameFlowWrapper.game.continue();
        } else {
             // Fallback to menuButton logic
             gameFlowWrapper.game.menuButton(playerId, arg, uuid, method);
             gameFlowWrapper.game.continue();
        }

        res.json(getState());
    } catch (e) {
        console.error(e);
        res.status(500).json({ error: e.message });
    }
});

app.get('/state', (req, res) => {
    try {
        if (!gameFlowWrapper) throw new Error("Game not initialized. Call /reset first.");
        res.json(getState());
    } catch (e) {
        res.status(500).json({ error: e.message });
    }
});

function getState() {
    const serializeCard = (card) => {
        if (!card) return null;
        return {
            uuid: card.uuid,
            internalName: card.internalName,
            zone: card.zone.name,
            power: card.power,
            hp: card.hp,
            damage: card.damage,
            exhausted: card.exhausted
        };
    };

    const serializePlayer = (p) => {
        return {
            id: p.id,
            name: p.name,
            base: serializeCard(p.base),
            leader: serializeCard(p.leader),
            hand: p.hand.map(serializeCard),
            deck: p.deck.map(serializeCard),
            discard: p.discard.map(serializeCard),
            resources: p.resources.map(serializeCard),
            readyResourceCount: p.readyResourceCount,
            exhaustedResourceCount: p.exhaustedResourceCount,
            spaceArena: p.spaceArena.map(serializeCard),
            groundArena: p.groundArena.map(serializeCard)
        };
    };

    return {
        phase: gameFlowWrapper.game.currentPhase,
        activePlayer: gameFlowWrapper.game.initiativePlayer?.id,
        player1Id: gameFlowWrapper.player1Id,
        player2Id: gameFlowWrapper.player2Id,
        prompts: {
            player1: gameFlowWrapper.player1.player.currentPrompt(),
            player2: gameFlowWrapper.player2.player.currentPrompt()
        },
        state: {
            player1: serializePlayer(gameFlowWrapper.player1.player),
            player2: serializePlayer(gameFlowWrapper.player2.player)
        }
    };
}

const PORT = 3000;
app.listen(PORT, () => {
    console.log(`RL Env Server listening on port ${PORT}`);
});
