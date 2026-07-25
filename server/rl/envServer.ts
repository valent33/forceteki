const express = require('express');
const GameFlowWrapper = require('../../test/helpers/GameFlowWrapper.js');
const { UnitTestCardDataGetter } = require('../../server/utils/cardData/UnitTestCardDataGetter.js');
const { UndoMode } = require('../../server/game/core/snapshot/SnapshotManager.js');

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
        const path = require('path');
        const directory = path.join(process.cwd(), 'test/json');
        const cardDataGetter = new UnitTestCardDataGetter(directory);

        const setCodeToInternalName = new Map();
        for (const [setCode, cardId] of cardDataGetter.setCodeMap.entries()) {
            const cardEntry = cardDataGetter.cardMap.get(cardId);
            if (cardEntry?.internalName) {
                setCodeToInternalName.set(setCode, cardEntry.internalName);
            }
        }

        gameFlowWrapper = new GameFlowWrapper(
            cardDataGetter,
            mockRouter,
            { id: '111', username: 'player1', settings: { optionSettings: { autoSingleTarget: false } } },
            { id: '222', username: 'player2', settings: { optionSettings: { autoSingleTarget: false } } },
            UndoMode.Disabled
        );

        const DeckBuilder = require('../../test/helpers/DeckBuilder.js');
        const deckBuilder = new DeckBuilder(cardDataGetter);

        const convertDeck = (deckInput, label) => {
            if (!deckInput) return null;

            const arr = [];
            const deckEntries = Array.isArray(deckInput)
                ? deckInput
                : Object.entries(deckInput).map(([cardId, count]) => ({ id: cardId, count }));

            for (const entry of deckEntries) {
                const cardId = typeof entry === 'string' ? entry : (entry.id || entry.card || entry.cardId);
                const count = typeof entry === 'string' ? 1 : Number(entry.count || 1);

                for (let i = 0; i < count; i++) {
                    arr.push(cardId);
                }

                const resolvedInternalName = setCodeToInternalName.get(cardId);
                if (resolvedInternalName === 'underworld-thug') {
                    console.log(`[envServer] ${label}: ${cardId} -> underworld-thug x${count}`);
                } else if (process.env.RL_DECK_DEBUG === '1' && resolvedInternalName && resolvedInternalName !== cardId) {
                    console.log(`[envServer] ${label}: ${cardId} -> ${resolvedInternalName} x${count}`);
                }
            }

            return arr.length > 0 ? arr : null;
        };

        const defaultDeck = [];

        const p1Options = {
            leader: req.body.options?.p1Leader || req.body.p1Leader || 'darth-vader#dark-lord-of-the-sith',
            base: req.body.options?.p1Base || req.body.p1Base || 'kestro-city',
            deck: convertDeck(req.body.options?.p1Cards || req.body.p1Cards, 'p1Cards') || defaultDeck
        };

        const p2Options = {
            leader: req.body.options?.p2Leader || req.body.p2Leader || 'luke-skywalker#faithful-friend',
            base: req.body.options?.p2Base || req.body.p2Base || 'administrators-tower',
            deck: convertDeck(req.body.options?.p2Cards || req.body.p2Cards, 'p2Cards') || defaultDeck
        };

        const player1OwnedCards = deckBuilder.getOwnedCards(1, p1Options, p2Options);
        const player2OwnedCards = deckBuilder.getOwnedCards(2, p2Options, p1Options);

        const [p1DeckObj] = deckBuilder.customDeck(1, player1OwnedCards, req.body.options?.phase || 'setup');
        const [p2DeckObj] = deckBuilder.customDeck(2, player2OwnedCards, req.body.options?.phase || 'setup');

        gameFlowWrapper.player1.selectDeck(p1DeckObj);
        gameFlowWrapper.player2.selectDeck(p2DeckObj);

        await gameFlowWrapper.startGameAsync();

        res.json(getState());
    } catch (e) {
        console.error(e);
        res.status(500).json({ error: e.message, stack: e.stack });
    }
});

app.post('/step', (req, res) => {
    try {
        if (!gameFlowWrapper) throw new Error("Game not initialized. Call /reset first.");
        
        const { playerId, action, arg, uuid, method, promptText, result, cardUuid } = req.body;
        
        // Find player interaction wrapper
        const playerWrapper = gameFlowWrapper.allPlayers.find(p => p.id === playerId);
        
        if (action === 'clickPrompt' && promptText) {
            // Some prompts need the UUID mapping directly mapped to game.menuButton from the wrapper.
            // playerWrapper.clickPrompt throws assertion errors trying to format debug strings if things mismatch in headless
            const promptButton = playerWrapper.player.currentPrompt().buttons.find((b: any) => 
                b.text.toString().toLowerCase() === promptText.toLowerCase()
            );
            if(promptButton) {
                 gameFlowWrapper.game.menuButton(playerId, promptButton.arg, promptButton.uuid, promptButton.method);
                 gameFlowWrapper.game.continue();
              } else if (playerWrapper.player.currentPrompt().dropdownListOptions?.includes(promptText)) {
                  gameFlowWrapper.game.menuButton(
                      playerId,
                      promptText,
                      playerWrapper.player.currentPrompt().promptUuid,
                      'menuButton'
                  );
                  gameFlowWrapper.game.continue();
            } else {
                 playerWrapper.clickPrompt(promptText); // fallback to throw default error
            }
           } else if (action === 'statefulPromptResults') {
               if (!result) {
                  throw new Error('Missing statefulPromptResults payload');
               }
               gameFlowWrapper.game.statefulPromptResults(playerId, result, uuid);
               gameFlowWrapper.game.continue();
        } else if (action === 'clickCard' && uuid) {
             gameFlowWrapper.game.cardClicked(playerId, uuid);
             gameFlowWrapper.game.continue();
        } else if (action === 'perCardMenuButton') {
             gameFlowWrapper.game.perCardMenuButton(playerId, arg, cardUuid, uuid, method);
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
        let damage = 0;
        try { damage = card.damage; } catch(e) {}
        let power = null;
        try { power = typeof card.getPower === 'function' ? card.getPower() : card.power; } catch(e) {}
        let hp = null;
        try { hp = typeof card.getHp === 'function' ? card.getHp() : card.hp; } catch(e) {}
        let exhausted = false;
        try { exhausted = card.exhausted; } catch(e) {}
        
        let upgradesArr = [];
        try { upgradesArr = (card.upgrades || []).map(u => ({ uuid: u.uuid, internalName: u.internalName })); } catch (e) { upgradesArr = []; }

        return {
            uuid: card.uuid,
            internalName: card.internalName,
            zone: card.zone?.name || 'unknown',
            power,
            hp,
            damage,
            exhausted,
            upgrades: upgradesArr
        };
    };

    const serializeArenaCards = (cards) => (cards || [])
        .filter((card) => {
            try {
                return !card.isAttached();
            } catch (e) {
                return true;
            }
        })
        .map(serializeCard);

    const serializePlayer = (pWrapper) => {
        const p = pWrapper.player;
        const hand = pWrapper.hand || [];
        const deck = p.deckZone?.cards || [];
        const discard = p.discardZone?.cards || [];
        const resources = p.resourceZone?.cards || [];
        const spaceArena = pWrapper.spaceArena || [];
        const groundArena = pWrapper.groundArena || [];
        
        return {
            id: p.id,
            name: p.name,
            base: serializeCard(p.base),
            leader: serializeCard(p.leader),
            hand: hand.map(serializeCard),
            deck: deck.map(serializeCard),
            discard: discard.map(serializeCard),
            resources: resources.map(serializeCard),
            readyResourceCount: p.readyResourceCount,
            exhaustedResourceCount: p.exhaustedResourceCount,
            hasForceToken: p.hasTheForce,
            credits: p.creditTokenCount,
            spaceArena: serializeArenaCards(spaceArena),
            groundArena: serializeArenaCards(groundArena)
        };
    };

    const p1State = gameFlowWrapper.player1.player.promptState;
    const p2State = gameFlowWrapper.player2.player.promptState;

    return {
        phase: gameFlowWrapper.game.currentPhase,
        activePlayer: gameFlowWrapper.game.initiativePlayer?.id,
        player1Id: gameFlowWrapper.player1Id,
        player2Id: gameFlowWrapper.player2Id,
        prompts: {
            player1: {
                ...gameFlowWrapper.player1.player.currentPrompt(),
                selectableCards: p1State?.selectableCards?.map((c: any) => c.uuid) || [],
                selectedCards: p1State?.selectedCards?.map((c: any) => c.uuid) || [],
                debug_legalActions: (gameFlowWrapper.player1.player as any).handZone.cards.map((c: any) => {
                    const player = gameFlowWrapper.player1.player;
                    const acts = c.getActions();
                    return {
                        id: c.internalName,
                        actions: acts.map((a:any) => ({
                            title: a.title,
                            req: a.meetsRequirements(a.createContext(player)),
                            isPlay: !!a.isPlayCardAction
                        }))
                    }
                })
            },
            player2: {
                ...gameFlowWrapper.player2.player.currentPrompt(),
                selectableCards: p2State?.selectableCards?.map((c: any) => c.uuid) || [],
                selectedCards: p2State?.selectedCards?.map((c: any) => c.uuid) || [],
                debug_legalActions: (gameFlowWrapper.player2.player as any).handZone.cards.map((c: any) => {
                    const player = gameFlowWrapper.player2.player;
                    const acts = c.getActions();
                    return {
                        id: c.internalName,
                        actions: acts.map((a:any) => ({
                            title: a.title,
                            req: a.meetsRequirements(a.createContext(player)),
                            isPlay: !!a.isPlayCardAction
                        }))
                    }
                })
            }
        },
        state: {
            player1: serializePlayer(gameFlowWrapper.player1),
            player2: serializePlayer(gameFlowWrapper.player2)
        }
    };
}

const PORT = 3005;
app.listen(PORT, () => {
    console.log(`RL Env Server listening on port ${PORT}`);
});
