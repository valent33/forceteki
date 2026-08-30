import type { IAbilityHelper } from '../../../AbilityHelper';
import type { INonLeaderUnitAbilityRegistrar } from '../../../core/card/AbilityRegistrationInterfaces';
import { NonLeaderUnitCard } from '../../../core/card/NonLeaderUnitCard';
import { RelativePlayer, TargetMode, WildcardCardType, ZoneName } from '../../../core/Constants';
import type { StateWatcherRegistrar } from '../../../core/stateWatcher/StateWatcherRegistrar';
import type { CardsLeftPlayThisPhaseWatcher } from '../../../stateWatchers/CardsLeftPlayThisPhaseWatcher';

export default class LattsRazziDeadlyWhipmaster extends NonLeaderUnitCard {
    private cardsLeftPlayThisPhase: CardsLeftPlayThisPhaseWatcher;

    protected override getImplementationId() {
        return {
            id: '3422504128',
            internalName: 'latts-razzi#deadly-whipmaster'
        };
    }

    protected override setupStateWatchers(registrar: StateWatcherRegistrar, AbilityHelper: IAbilityHelper): void {
        this.cardsLeftPlayThisPhase = AbilityHelper.stateWatchers.cardsLeftPlayThisPhase();
    }

    public override setupCardAbilities(registrar: INonLeaderUnitAbilityRegistrar, AbilityHelper: IAbilityHelper) {
        registrar.addWhenPlayedAbility({
            title: 'Give a Shield token or an Experience token to this unit. Then, she deals damage equal to her power to an enemy ground unit',
            targetResolver: {
                mode: TargetMode.Select,
                choices: {
                    ['Give a Shield Token to this unit']: AbilityHelper.immediateEffects.giveShield(),
                    ['Give an Experience token to this unit']: AbilityHelper.immediateEffects.giveExperience(),
                }
            },
            then: (thenContext) => {
                // The unit may have left play while the shield/experience
                // choice resolved — getPower() contract-asserts on a card in
                // discard (via its `upgrades` zone check), so guard it.
                const power = thenContext.source.isInPlay() ? thenContext.source.getPower() : 0;
                return {
                    title: 'Deal damage equal to her power to an enemy ground unit',
                    targetResolver: {
                        activePromptTitle: `Deal ${power} damage to an enemy ground unit`,
                        controller: RelativePlayer.Opponent,
                        cardTypeFilter: WildcardCardType.Unit,
                        zoneFilter: ZoneName.GroundArena,
                        immediateEffect: AbilityHelper.immediateEffects.damage({
                            amount: power,
                            source: thenContext.source,
                        })
                    }
                };
            }
        });
    }
}