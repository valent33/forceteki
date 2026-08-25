import type { IAbilityHelper } from '../../../AbilityHelper';
import type { INonLeaderUnitAbilityRegistrar } from '../../../core/card/AbilityRegistrationInterfaces';
import { NonLeaderUnitCard } from '../../../core/card/NonLeaderUnitCard';
import { AbilityType } from '../../../core/Constants';

export default class IdenVersioAdaptOrDie extends NonLeaderUnitCard {
    protected override getImplementationId () {
        return {
            id: '9325037410',
            internalName: 'iden-versio#adapt-or-die',
        };
    }

    public override setupCardAbilities(registrar: INonLeaderUnitAbilityRegistrar, AbilityHelper: IAbilityHelper) {
        registrar.addPilotingAbility({
            type: AbilityType.Triggered,
            title: 'Give a Shield token to attached unit',
            when: {
                onUpgradeAttached: (event, context) => event.upgradeCard === context.source
            },
            immediateEffect: AbilityHelper.immediateEffects.giveShield((context) => ({
                // If the pilot was detached before this queued trigger resolves
                // (e.g. its attached unit left play), `parentCard` is null and the
                // getter contract-asserts — so guard with isInPlay() like
                // PerilousPosition does instead of relying on `??`.
                target: context.source.isInPlay() ? context.source.parentCard : null
            }))
        });
    }
}